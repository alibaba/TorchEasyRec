# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Unit tests for ``tzrec.acc.aot_utils._build_aoti_output_field_names``.

These tests verify that the output-name alignment helper correctly maps the
eager forward's return-dict keys onto the AOTI output-handle layout emitted
by ``torch.export.export``. The helper is pure Python and only reads
``exported_pg.graph_signature.output_specs[i].kind`` / ``.target``, so the
tests use lightweight duck-typed fakes instead of invoking the real
``torch.export`` machinery. That keeps the tests fast, deterministic, and
independent of the surrounding CUDA / Triton / TorchRec stack.

Regression coverage:

* ``test_user_output_only``: the original (pre-bugfix) happy path where
  ``USER_OUTPUT`` is the only output kind.
* ``test_mixed_output_kinds``: the buggy scenario from the HSTU export that
  motivated this helper. Non-``USER_OUTPUT`` slots must be filled with
  placeholders so the JSON written to disk has one entry per AOTI output
  handle (and downstream tensors are not renamed by position drift).
* ``test_user_output_count_mismatch_raises``: if the exported program does
  not expose the same number of USER_OUTPUT slots as the eager dict returns,
  the helper must refuse to emit a mislabeled mapping.
* ``test_string_kind_fallback``: older torch builds surface ``kind`` as a
  bare string rather than an enum; the helper must still identify
  ``USER_OUTPUT`` slots in that case.
"""

import unittest
from typing import Any, List, Optional

from tzrec.acc.aot_utils import _build_aoti_output_field_names


class _FakeKind:
    """Duck-typed stand-in for ``torch.export.graph_signature.OutputKind``.

    The real enum exposes ``.name`` (e.g. ``"USER_OUTPUT"``,
    ``"BUFFER_MUTATION"``). ``_build_aoti_output_field_names`` reads exactly
    that attribute plus ``str(kind)``, nothing else.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # matches enum ``str(OutputKind.X)`` shape.
        return f"OutputKind.{self.name}"


class _FakeSpec:
    """Stand-in for one entry of ``graph_signature.output_specs``."""

    def __init__(self, kind: Any, target: Optional[str] = None) -> None:
        self.kind = kind
        self.target = target


class _FakeSignature:
    def __init__(self, output_specs: List[_FakeSpec]) -> None:
        self.output_specs = output_specs


class _FakeExportedProgram:
    """Minimal duck-typed ExportedProgram exposing only ``.graph_signature``."""

    def __init__(self, output_specs: List[_FakeSpec]) -> None:
        self.graph_signature = _FakeSignature(output_specs)


def _make_program(kinds: List[str]) -> _FakeExportedProgram:
    """Build a fake ExportedProgram with the given sequence of kind names."""
    return _FakeExportedProgram([_FakeSpec(_FakeKind(name)) for name in kinds])


class BuildAotiOutputFieldNamesTest(unittest.TestCase):
    def test_user_output_only(self) -> None:
        """All outputs are USER_OUTPUT: names pass through unchanged."""
        program = _make_program(["USER_OUTPUT", "USER_OUTPUT", "USER_OUTPUT"])
        names = _build_aoti_output_field_names(program, ["logits", "probs", "length"])
        self.assertEqual(names, ["logits", "probs", "length"])

    def test_mixed_output_kinds(self) -> None:
        """Regression: USER_OUTPUT slots may appear after extra output kinds.

        Mirrors the HSTU export that motivated this helper: 35 Inductor-emitted
        slots (parameter-preprocessing / buffer-mutation / token outputs)
        precede the 5 actual USER_OUTPUT slots. Eager keys must land on the
        USER_OUTPUT positions only, with placeholders elsewhere.
        """
        kinds = (
            ["BUFFER_MUTATION"] * 3
            + ["TOKEN"]
            + ["USER_INPUT_MUTATION"] * 2
            + ["USER_OUTPUT"] * 3
        )
        program = _make_program(kinds)
        eager_keys = ["logits_is_click", "probs_is_click", "length"]

        names = _build_aoti_output_field_names(program, eager_keys)

        # One name per AOTI output handle.
        self.assertEqual(len(names), len(kinds))
        # USER_OUTPUT slots receive eager keys in order.
        self.assertEqual(names[-3:], eager_keys)
        # Non-USER_OUTPUT slots are all placeholders (never eager keys).
        for i in range(6):
            self.assertTrue(
                names[i].startswith("_unused_"),
                f"expected placeholder at position {i}, got {names[i]!r}",
            )
            # Placeholder carries its index so collisions are impossible.
            self.assertIn(f"_{i}_", names[i])
        # And no placeholder ever duplicates a real user-output name.
        for placeholder in names[:6]:
            self.assertNotIn(placeholder, eager_keys)

    def test_placeholders_are_unique(self) -> None:
        """Placeholders embed their slot index so names are globally unique."""
        kinds = ["BUFFER_MUTATION"] * 4 + ["USER_OUTPUT"]
        program = _make_program(kinds)
        names = _build_aoti_output_field_names(program, ["out"])
        self.assertEqual(len(set(names)), len(names))

    def test_user_output_count_mismatch_raises(self) -> None:
        """Fail loudly rather than silently misalign names."""
        program = _make_program(["USER_OUTPUT", "USER_OUTPUT"])
        with self.assertRaises(RuntimeError) as cm:
            _build_aoti_output_field_names(program, ["only_one_key"])
        msg = str(cm.exception)
        self.assertIn("USER_OUTPUT", msg)
        self.assertIn("only_one_key", msg)

    def test_string_kind_fallback(self) -> None:
        """Older torch versions may surface ``kind`` as a plain string."""
        # Mix of bare strings and enum-like objects to exercise both paths.
        specs = [
            _FakeSpec("OutputKind.BUFFER_MUTATION"),
            _FakeSpec("OutputKind.USER_OUTPUT"),
            _FakeSpec(_FakeKind("USER_OUTPUT")),
        ]
        program = _FakeExportedProgram(specs)
        names = _build_aoti_output_field_names(program, ["a", "b"])
        self.assertEqual(len(names), 3)
        self.assertTrue(names[0].startswith("_unused_0_"))
        self.assertEqual(names[1], "a")
        self.assertEqual(names[2], "b")

    def test_empty_outputs(self) -> None:
        """Zero outputs on both sides is a valid (if unusual) configuration."""
        program = _make_program([])
        names = _build_aoti_output_field_names(program, [])
        self.assertEqual(names, [])

    def test_all_non_user_output_with_nonempty_eager_raises(self) -> None:
        """If the graph exposes no USER_OUTPUT, any eager key is a mismatch."""
        program = _make_program(["BUFFER_MUTATION", "TOKEN"])
        with self.assertRaises(RuntimeError):
            _build_aoti_output_field_names(program, ["x"])


if __name__ == "__main__":
    unittest.main()
