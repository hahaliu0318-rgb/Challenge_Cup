from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
from pydantic import ValidationError

from app.gpu import GPUInfo, choose_gpu_candidate
from app.image_probe import ImageValidationError, validate_and_probe_images
from app.jobs import JobService
from app.output_parser import augment_prompt, extract_bbox, extract_choice, extract_choices, extract_number, parse_output
from app.routing import choose_route
from app.schemas import InferRequest
from app.task_classifier import TaskDecision, classify_by_rules, parse_model_classification
from app.worker_manager import WorkerManager, WorkerSlot
from workers.geollava_worker import build_record as build_geollava_record


class TaskClassifierTests(unittest.TestCase):
    def test_english_rules(self):
        self.assertEqual(classify_by_rules("How many ships are visible?", 1).type, "counting")
        self.assertEqual(classify_by_rules("What color is the roof?", 1).type, "color")
        self.assertEqual(classify_by_rules("Locate the aircraft and return its bounding box", 1).type, "grounding")

    def test_chinese_rules(self):
        self.assertEqual(classify_by_rules("请描述这张遥感图像", 1).type, "caption")
        self.assertEqual(classify_by_rules("图中有多少艘船？", 1).type, "counting")
        self.assertEqual(classify_by_rules("目标位于图像什么位置？", 1).type, "position")

    def test_two_images_force_change(self):
        decision = classify_by_rules("What happened?", 2)
        self.assertEqual(decision.type, "change_caption")

    def test_explicit_vqa_keeps_local_subtask(self):
        decision = classify_by_rules("How many cars?", 1, "vqa")
        self.assertEqual(decision.type, "vqa")
        self.assertEqual(decision.subtask, "counting")

    def test_explicit_task_controls_its_subtask(self):
        spatial = classify_by_rules("What is the position of the object?", 1, "spatial_relationship")
        self.assertEqual(spatial.subtask, "spatial_relationship")
        motion = classify_by_rules("Locate the moving aircraft.", 1, "motion_state")
        self.assertEqual(motion.subtask, "motion")
        object_task = classify_by_rules("Locate and classify the object.", 1, "object_classification")
        self.assertEqual(object_task.subtask, "object")

    def test_model_classification_json(self):
        decision = parse_model_classification('prefix {"task":"color","subtask":"color","confidence":0.87} suffix')
        self.assertIsNotNone(decision)
        self.assertEqual(decision.type, "color")

    def test_explicit_alias_and_invalid_task(self):
        self.assertEqual(classify_by_rules("question", 1, "count").type, "counting")
        with self.assertRaises(ValueError):
            classify_by_rules("question", 1, "unsupported_task")

    def test_low_confidence_fallback(self):
        decision = classify_by_rules("Please inspect this sample.", 1)
        self.assertEqual(decision.type, "vqa")
        self.assertLess(decision.confidence, 0.8)

    def test_complex_reasoning_and_classification_rules(self):
        self.assertEqual(classify_by_rules("Why is this route anomalous?", 1).type, "complex_reasoning")
        self.assertEqual(classify_by_rules("Classify the land use", 1).type, "land_use_classification")

    def test_invalid_model_classification(self):
        self.assertIsNone(parse_model_classification('{"task":"not_real","confidence":2}'))
        self.assertIsNone(parse_model_classification("not-json"))

    def test_model_subtask_is_controlled(self):
        free_text = parse_model_classification(
            '{"task":"caption","subtask":"Describe the object","confidence":0.8}'
        )
        self.assertIsNotNone(free_text)
        self.assertIsNone(free_text.subtask)
        counting = parse_model_classification('{"task":"counting","subtask":null,"confidence":0.9}')
        self.assertEqual(counting.subtask, "counting")


class RoutingTests(unittest.TestCase):
    @staticmethod
    def meta(edge: int) -> list[dict]:
        return [{"width": edge, "height": edge, "max_edge": edge, "pixels": edge * edge}]

    def test_1024_boundary(self):
        qwen = choose_route(TaskDecision("caption", None, "explicit", 1), self.meta(1024))
        geo = choose_route(TaskDecision("caption", None, "explicit", 1), self.meta(1025))
        self.assertEqual(qwen.worker, "qwen")
        self.assertEqual(geo.worker, "geollava")

    def test_high_resolution_local_detail(self):
        for task in ("counting", "color", "position", "grounding", "detection", "spatial_relationship"):
            with self.subTest(task=task):
                route = choose_route(TaskDecision(task, None, "explicit", 1), self.meta(4096))
                self.assertEqual(route.worker, "zoomsearch")

    def test_vqa_subtask_routes_zoom(self):
        route = choose_route(TaskDecision("vqa", "color", "explicit", 1), self.meta(4096))
        self.assertEqual(route.worker, "zoomsearch")

    def test_two_images_route_qwen(self):
        meta = self.meta(8000) + self.meta(8000)
        route = choose_route(TaskDecision("change_caption", None, "explicit", 1), meta)
        self.assertEqual(route.worker, "qwen")

    def test_high_resolution_semantic_tasks(self):
        for task in ("caption", "vqa", "scene_classification", "land_use_classification", "complex_reasoning"):
            with self.subTest(task=task):
                route = choose_route(TaskDecision(task, None, "explicit", 1), self.meta(4096))
                self.assertEqual(route.worker, "geollava")


class OutputParserTests(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(extract_choice("The answer is (C)."), "C")
        self.assertEqual(extract_number("There are five ships."), 5)
        self.assertEqual(extract_bbox('{"bbox_2d":[100,200,300,400]}'), [100, 200, 300, 400])

    def test_mcq_output(self):
        task = TaskDecision("vqa", None, "explicit", 1)
        answer, ok, _ = parse_output("Answer: B", task, "Question (A) one (B) two", [])
        self.assertTrue(ok)
        self.assertEqual(answer, "B")

    def test_count_and_bbox_output(self):
        count, count_ok, _ = parse_output(
            "There are 12 ships.", TaskDecision("counting", None, "explicit", 1), "How many?", []
        )
        self.assertTrue(count_ok)
        self.assertEqual(count, 12)
        bbox, bbox_ok, _ = parse_output(
            "[100, 200, 300, 400]",
            TaskDecision("grounding", None, "explicit", 1),
            "Locate it",
            [{"width": 1000, "height": 1000}],
        )
        self.assertTrue(bbox_ok)
        self.assertEqual(bbox, {"bbox_2d": [100, 200, 300, 400]})

    def test_bbox_normalization(self):
        self.assertEqual(extract_bbox("[0.1, 0.2, 0.3, 0.4]"), [100, 200, 300, 400])
        self.assertEqual(
            extract_bbox("[1000, 500, 3000, 1500]", {"width": 4000, "height": 2000}),
            [250, 250, 750, 750],
        )

    def test_parse_failures_are_explicit(self):
        answer, ok, warnings = parse_output(
            "unknown", TaskDecision("counting", None, "explicit", 1), "How many?", []
        )
        self.assertFalse(ok)
        self.assertEqual(answer, "unknown")
        self.assertIn("count_parse_failed", warnings)

    def test_prompt_augmentation_is_idempotent(self):
        prompt = "Question\nA. one\nB. two\nRespond with only the letter (A, B, C, D, or E)."
        self.assertEqual(augment_prompt(prompt, TaskDecision("vqa", None, "explicit", 1)), prompt)

    def test_xlrs_multi_choice_output(self):
        prompt = (
            "Question\nA. one\nB. two\nC. three\nD. four\n"
            "There may be more than one correct option. Only respond with the letter(s)."
        )
        self.assertEqual(extract_choices("The answers are (A) and C."), ["A", "C"])
        answer, ok, _ = parse_output(
            "A C", TaskDecision("land_use_classification", None, "explicit", 1), prompt, []
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "A C")
        self.assertEqual(augment_prompt(prompt, TaskDecision("land_use_classification", None, "explicit", 1)), prompt)


class ImageProbeTests(unittest.TestCase):
    def test_probe_and_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (512, 256), "white").save(image_path)
            paths, metadata = validate_and_probe_images([str(image_path)], [root])
            self.assertEqual(paths, [str(image_path.resolve())])
            self.assertEqual(metadata[0]["max_edge"], 512)

    def test_reject_outside_root(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            image_path = Path(outside) / "sample.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images([str(image_path)], [allowed])

    def test_reject_relative_directory_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images(["relative.png"], [root])
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images([str(root)], [root])
            corrupt = root / "bad.png"
            corrupt.write_bytes(b"not an image")
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images([str(corrupt)], [root])

    def test_reject_symlink_escape_when_supported(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "sample.png"
            Image.new("RGB", (10, 10), "white").save(target)
            link = Path(allowed) / "link.png"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is not available")
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images([str(link)], [allowed])

    def test_reject_pixel_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            with self.assertRaises(ImageValidationError):
                validate_and_probe_images([str(image_path)], [root], max_image_pixels=99)


class SchemaTests(unittest.TestCase):
    def test_three_fields_and_two_images(self):
        request = InferRequest(image=["/a.png", "/b.png"], text=" compare ", task_type="AUTO")
        self.assertEqual(request.text, "compare")
        self.assertIsNone(request.task_type)
        self.assertEqual(len(request.image_paths()), 2)

    def test_invalid_image_counts_and_blank_text(self):
        for images in ([], ["/a", "/b", "/c"]):
            with self.subTest(images=images), self.assertRaises(ValidationError):
                InferRequest(image=images, text="question", task_type=None)
        with self.assertRaises(ValidationError):
            InferRequest(image="/a", text="   ", task_type=None)

    def test_extra_api_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            InferRequest.model_validate({"image": "/a", "text": "q", "task_type": None, "dataset": "vrs"})

    def test_path_and_task_length_limits(self):
        with self.assertRaises(ValidationError):
            InferRequest(image="/" + "a" * 4097, text="q", task_type=None)
        with self.assertRaises(ValidationError):
            InferRequest(image="/a", text="q", task_type="x" * 65)


class GPUSelectionTests(unittest.TestCase):
    @staticmethod
    def inventory() -> list[GPUInfo]:
        return [
            GPUInfo(0, "GPU0", 24576, 24000, 576, 0),
            GPUInfo(1, "GPU1", 24576, 12000, 12576, 0),
            GPUInfo(2, "GPU2", 10240, 9900, 340, 0),
        ]

    def test_selects_first_satisfying_candidate(self):
        candidate = choose_gpu_candidate(self.inventory(), [[1, 2], [0, 2]], [23000, 9500], {})
        self.assertEqual(candidate, [0, 2])

    def test_rejects_leased_or_insufficient_gpu(self):
        candidate = choose_gpu_candidate(self.inventory(), [[0, 2]], [23000, 9500], {0: "qwen"})
        self.assertIsNone(candidate)


class WorkerManagerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def config() -> dict:
        return {
            "runtime": {"worker_idle_ttl_sec": 900, "gpu_poll_interval_sec": 15},
            "models": {
                "qwen": {"enabled": True},
                "change_agent": {"enabled": False, "phase": 2, "disabled_reason": "phase 2"},
            },
        }

    async def test_disabled_worker_is_not_schedulable(self):
        manager = WorkerManager(self.config())
        self.assertIn("qwen", manager.model_configs)
        self.assertNotIn("change_agent", manager.model_configs)
        self.assertEqual(manager.disabled_models["change_agent"]["phase"], 2)

    async def test_idle_reaper_stops_expired_worker(self):
        manager = WorkerManager(self.config())
        manager.slots["qwen"] = WorkerSlot(
            name="qwen",
            process=SimpleNamespace(returncode=None, pid=123),
            gpus=[0],
            load_seconds=1.0,
            started_at=0.0,
            last_used_at=0.0,
        )
        manager.stop_worker = AsyncMock()
        await manager._reap_idle_once(now=901.0)
        manager.stop_worker.assert_awaited_once_with("qwen")

    async def test_startup_cancellation_releases_reserved_lease(self):
        config = self.config()
        config["models"]["qwen"].update(
            {"python": "/unused", "worker_script": "/unused", "working_dir": "/unused", "env": {}}
        )
        manager = WorkerManager(config)

        async def reserve(*_args, **_kwargs):
            manager.leases[0] = "qwen"
            return [0]

        async def cancel_during_status(_status):
            raise asyncio.CancelledError()

        manager._reserve_gpus = reserve
        with self.assertRaises(asyncio.CancelledError):
            await manager._start_slot("qwen", cancel_during_status, None)
        self.assertEqual(manager.leases, {})


class JobServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_survives_caller_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            service = JobService(
                {
                    "runtime": {
                        "jobs_db": str(Path(directory) / "jobs.sqlite3"),
                        "job_retention_days": 7,
                    }
                },
                AsyncMock(),
            )
            started = asyncio.Event()
            release = asyncio.Event()
            finished = asyncio.Event()

            async def delayed_submit(_request):
                started.set()
                await release.wait()
                finished.set()
                return "durable-job-id"

            service.submit = delayed_submit
            caller = asyncio.create_task(
                service.submit_durable(InferRequest(image="/unused.png", text="question"))
            )
            await started.wait()
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            self.assertEqual(len(service.submission_tasks), 1)

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(service.submission_tasks, set())
            service.store.close()


class WorkerContractTests(unittest.TestCase):
    def test_geollava_record_satisfies_existing_question_builder(self):
        record = build_geollava_record(
            {"id": "request-id", "text": "Question with choices already included", "images": ["/a.png"]}
        )
        self.assertEqual(record["multi-choice options"], "")
        self.assertEqual(record["image_paths"], ["/a.png"])
        self.assertEqual(record["question"], "Question with choices already included")


if __name__ == "__main__":
    unittest.main()
