"""What a segment declares it needs, and whether a worker has it (console#422).

The bug these prevent is silent in both directions. Extract too little and a pod claims a
render it cannot load, which is the failure this replaced. Extract too much — a requirement
for a LoRA called "none", say — and a segment nothing can satisfy sits PENDING forever
beside an idle fleet, which looks exactly like an empty queue.
"""
from app.ltx_stack import LTX_STACK
from app.model_requirements import (
    CHECKPOINT,
    LORA,
    Artifact,
    canonical,
    describe,
    required_artifacts,
    unsatisfied,
)


class TestWhatARecipeDeclares:
    def test_the_checkpoint_is_read_from_the_field_not_the_name(self):
        """The entire point of the ticket: no code may infer a model from a recipe or file
        name. A heuristic that recognises 10Eros_v1.5_bf16 misses 10Eros_v2 as a PASS."""
        r = required_artifacts({"recipe": "Doggystyle 10Eros edition",
                                "checkpoint": "sulphur_dev_bf16"})
        assert r == {Artifact(CHECKPOINT, "sulphur_dev_bf16")}

    def test_a_recipe_without_a_checkpoint_requires_the_stack_default(self):
        """Segments predating per-pose base models. They render on the stack's checkpoint, so
        that is what they require — resolving it here keeps three call sites from each
        growing their own `or LTX_STACK[...]`."""
        assert required_artifacts({"char_lora": "none"}) == {
            Artifact(CHECKPOINT, canonical(LTX_STACK["checkpoint"]))
        }

    def test_character_and_content_loras_are_requirements_too(self):
        r = required_artifacts({
            "checkpoint": "sulphur_dev_bf16",
            "char_lora": "k3llydw_v2",
            "content_loras": [{"name": "d0ggyff", "s1": 0.8}, {"name": "SexGod", "s1": 0.6}],
        })
        assert r == {
            Artifact(CHECKPOINT, "sulphur_dev_bf16"),
            Artifact(LORA, "k3llydw_v2"),
            Artifact(LORA, "d0ggyff"),
            Artifact(LORA, "SexGod"),
        }

    def test_none_is_a_choice_not_a_file(self):
        """"none" is a real value here — the CHARACTER dropdown offers it and the stack's
        content_lora IS the string "none". Requiring none.safetensors would block every
        segment that chose to run without one."""
        r = required_artifacts({"checkpoint": "c", "char_lora": "none",
                                "content_loras": [{"name": "None"}]})
        assert r == {Artifact(CHECKPOINT, "c")}

    def test_the_two_spellings_of_a_file_are_one_requirement(self):
        """Recipes store a bare stem, ComfyUI names files. If those ever drift apart the gate
        does not fail loudly — it matches nothing and the worker claims no work."""
        assert required_artifacts({"checkpoint": "sulphur_dev_bf16.safetensors"}) == \
               required_artifacts({"checkpoint": "sulphur_dev_bf16"})

    def test_a_non_recipe_segment_requires_nothing(self):
        """NULL ltx_recipe is a WAN segment, a free-form LTX one, or a CPU reprocess carrier.
        Declaring nothing must mean requiring nothing — this may not withhold work that
        already flows."""
        assert required_artifacts(None) == set()
        assert required_artifacts({}) == set()

    def test_the_older_content_lora_shape_does_not_raise(self):
        """Segments rendered before content LoRAs became objects are still in the database and
        still re-rollable. A KeyError here would take down a job page."""
        assert Artifact(LORA, "old") in required_artifacts(
            {"checkpoint": "c", "content_loras": ["old"]}
        )


class TestWhatAWorkerCanRun:
    CKPT = {CHECKPOINT: {"sulphur_dev_bf16", "ltx-2.3-22b-dev"}}

    def test_a_missing_checkpoint_is_named(self):
        r = required_artifacts({"checkpoint": "10Eros_v1.5_bf16"})
        assert unsatisfied(r, self.CKPT, ["lora"]) == {Artifact(CHECKPOINT, "10Eros_v1.5_bf16")}

    def test_a_checkpoint_it_holds_satisfies(self):
        r = required_artifacts({"checkpoint": "sulphur_dev_bf16"})
        assert unsatisfied(r, self.CKPT, ["lora"]) == set()

    def test_loras_never_gate_because_the_worker_fetches_them(self):
        """The daemon downloads a LoRA a pose names but this worker has never seen. Gating on
        one would refuse work the worker would have handled by itself."""
        r = required_artifacts({"checkpoint": "sulphur_dev_bf16", "char_lora": "brand_new"})
        assert unsatisfied(r, self.CKPT, ["lora"]) == set()

    def test_a_worker_that_learns_to_fetch_checkpoints_stops_being_gated(self):
        """console#423's whole seam: the daemon says "checkpoint" and this opens, with no API
        change and no coordinated deploy."""
        r = required_artifacts({"checkpoint": "10Eros_v1.5_bf16"})
        assert unsatisfied(r, self.CKPT, ["lora", "checkpoint"]) == set()

    def test_an_unreported_kind_does_not_gate(self):
        """An older daemon reports no checkpoints at all. Starving it on upgrade day would be
        a worse failure than the one being fixed, and it would look like a dead queue."""
        r = required_artifacts({"checkpoint": "10Eros_v1.5_bf16"})
        assert unsatisfied(r, {}, []) == set()

    def test_a_worker_reporting_an_empty_list_is_not_the_same_as_never_reporting(self):
        r = required_artifacts({"checkpoint": "10Eros_v1.5_bf16"})
        assert unsatisfied(r, {CHECKPOINT: set()}, []) == {Artifact(CHECKPOINT, "10Eros_v1.5_bf16")}

    def test_either_spelling_in_the_inventory_satisfies(self):
        r = required_artifacts({"checkpoint": "sulphur_dev_bf16"})
        assert unsatisfied(r, {CHECKPOINT: {"sulphur_dev_bf16.safetensors"}}, []) == set()

    def test_the_message_names_the_kind_and_the_file(self):
        """It is read off a stalled job page by someone deciding what to do next."""
        assert describe({Artifact(CHECKPOINT, "10Eros_v1.5_bf16")}) == \
               "checkpoint '10Eros_v1.5_bf16'"
