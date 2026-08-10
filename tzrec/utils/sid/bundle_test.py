# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import unittest

from tzrec.utils.sid import bundle
from tzrec.utils.test_util import make_test_dir


def _manifest(**kw):
    base = dict(
        bundle_uuid="uuid-v2",
        codebook=[8, 8],
        capacity=5,
        max_observed_items_per_sid=3,
        artifacts={
            bundle.SID_TO_ITEMS_ARTIFACT: bundle.ArtifactEntry(
                save_type="parquet",
                rows=4,
                location="./sid/v2/sid_to_items",
            )
        },
    )
    base.update(kw)
    return bundle.BundleManifest(**base)


class BundleLocationTest(unittest.TestCase):
    def test_odps_root_tolerates_a_trailing_slash(self) -> None:
        self.assertEqual(
            bundle.artifact_location(
                "odps://prj/tables/sid/", "v1", bundle.ITEM_TO_SID_ARTIFACT
            ),
            "odps://prj/tables/sid/generation=v1/artifact=item_to_sid",
        )

    def test_odps_read_pattern_is_the_partition_itself(self) -> None:
        self.assertEqual(
            bundle.artifact_read_pattern(
                "odps://prj/tables/sid", "v1", bundle.ITEM_TO_SID_ARTIFACT, "parquet"
            ),
            "odps://prj/tables/sid/generation=v1/artifact=item_to_sid",
        )


class BundleLayoutTest(unittest.TestCase):
    def _layout(self, **kw):
        base = dict(
            output_path="./sid",
            item_to_sid_root="./map",
            generation="v2",
            from_generation="v1",
        )
        base.update(kw)
        return bundle.BundleLayout(**base)

    def test_each_family_resolves_under_its_own_root(self) -> None:
        layout = self._layout()
        self.assertEqual(
            layout.write_path(bundle.ITEM_TO_SID_ARTIFACT), "./map/v2/item_to_sid"
        )
        self.assertEqual(
            layout.write_path(bundle.DELTA_ITEM_TO_SID_ARTIFACT),
            "./map/v2/delta_item_to_sid",
        )
        self.assertEqual(
            layout.write_path(bundle.SID_TO_ITEMS_ARTIFACT), "./sid/v2/sid_to_items"
        )
        self.assertEqual(
            layout.write_path(bundle.DELTA_SID_TO_ITEMS_ARTIFACT),
            "./sid/v2/delta_sid_to_items",
        )

    def test_reads_come_from_the_appended_onto_generation(self) -> None:
        layout = self._layout()
        self.assertEqual(
            layout.read_path(bundle.ITEM_TO_SID_ARTIFACT, "parquet"),
            "./map/v1/item_to_sid/part-*.parquet",
        )
        self.assertEqual(
            layout.read_path(bundle.SID_TO_ITEMS_ARTIFACT, "parquet"),
            "./sid/v1/sid_to_items/part-*.parquet",
        )
        self.assertEqual(
            layout.read_path(bundle.ITEM_TO_SID_ARTIFACT, "csv"),
            "./map/v1/item_to_sid/part-*.csv",
        )

    def test_an_odps_item_to_sid_root_leaves_the_serving_set_on_files(self) -> None:
        layout = self._layout(item_to_sid_root="odps://prj/tables/sid")
        self.assertEqual(
            layout.write_path(bundle.ITEM_TO_SID_ARTIFACT),
            "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
        )
        self.assertEqual(
            layout.write_path(bundle.SID_TO_ITEMS_ARTIFACT), "./sid/v2/sid_to_items"
        )

    def test_manifest_locations_track_the_two_generations(self) -> None:
        layout = self._layout()
        self.assertEqual(layout.prior_manifest_path(), "./sid/v1/manifest.json")

    def test_a_missing_root_is_reported_against_its_artifact(self) -> None:
        layout = self._layout(item_to_sid_root=None)
        with self.assertRaisesRegex(RuntimeError, "item_to_sid"):
            layout.write_path(bundle.ITEM_TO_SID_ARTIFACT)
        self.assertEqual(
            layout.write_path(bundle.SID_TO_ITEMS_ARTIFACT), "./sid/v2/sid_to_items"
        )

    def test_an_odps_output_path_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "always files"):
            bundle.BundleLayout(
                output_path="odps://prj/tables/g",
                item_to_sid_root="./map",
                generation="v2",
                from_generation=None,
            )

    def test_a_full_resolve_has_nothing_to_read(self) -> None:
        layout = self._layout(from_generation=None)
        with self.assertRaisesRegex(RuntimeError, "from_generation"):
            layout.read_path(bundle.SID_TO_ITEMS_ARTIFACT, "parquet")


class BundleManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def test_round_trip_preserves_every_field(self) -> None:
        original = _manifest(since_bundle_uuid="uuid-v1", item_id_type="int64")
        restored = bundle.BundleManifest.from_json(original.to_json())
        self.assertEqual(restored, original)

    def test_absent_optional_fields_are_omitted_not_null(self) -> None:
        payload = json.loads(_manifest().to_json())
        self.assertNotIn("since_bundle_uuid", payload)
        self.assertIn("location", payload["artifacts"][bundle.SID_TO_ITEMS_ARTIFACT])

    def test_rejects_a_manifest_missing_required_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            bundle.BundleManifest.from_json(json.dumps({"bundle_uuid": "x"}))

    def test_rejects_a_manifest_without_the_observed_ceiling(self) -> None:
        payload = json.loads(_manifest().to_json())
        del payload["max_observed_items_per_sid"]
        with self.assertRaisesRegex(ValueError, "max_observed_items_per_sid"):
            bundle.BundleManifest.from_json(json.dumps(payload))

    def test_an_odps_entry_keeps_the_url_its_readers_take(self) -> None:
        layout = bundle.BundleLayout(
            output_path="./sid",
            item_to_sid_root="odps://prj/tables/sid",
            generation="v2",
            from_generation=None,
        )
        entry = layout.entry(bundle.ITEM_TO_SID_ARTIFACT, 9, "ParquetWriter")
        self.assertEqual(entry.save_type, "odps")
        self.assertEqual(
            entry.location,
            "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
        )

    def test_a_file_entry_keeps_the_writer_format(self) -> None:
        layout = bundle.BundleLayout(
            output_path="./sid",
            item_to_sid_root="./map",
            generation="v2",
            from_generation=None,
        )
        entry = layout.entry(bundle.ITEM_TO_SID_ARTIFACT, 4, "CsvWriter")
        self.assertEqual(entry.save_type, "csv")
        self.assertEqual(entry.location, "./map/v2/item_to_sid")

    def test_a_csv_writer_never_makes_the_serving_set_csv(self) -> None:
        layout = bundle.BundleLayout(
            output_path="./sid",
            item_to_sid_root="./map",
            generation="v2",
            from_generation=None,
        )
        self.assertEqual(
            layout.entry(bundle.SID_TO_ITEMS_ARTIFACT, 4, "CsvWriter").save_type,
            "parquet",
        )

    def test_rejects_a_serving_artifact_that_is_not_parquet(self) -> None:
        payload = json.loads(_manifest().to_json())
        payload["artifacts"][bundle.SID_TO_ITEMS_ARTIFACT]["save_type"] = "odps"
        with self.assertRaisesRegex(ValueError, "serving set is always parquet"):
            bundle.BundleManifest.from_json(json.dumps(payload))

    def test_item_to_sid_artifact_may_be_odps(self) -> None:
        payload = json.loads(_manifest().to_json())
        payload["artifacts"][bundle.ITEM_TO_SID_ARTIFACT] = {
            "save_type": "odps",
            "rows": 9,
            "location": "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
        }
        restored = bundle.BundleManifest.from_json(json.dumps(payload))
        self.assertEqual(
            restored.artifacts[bundle.ITEM_TO_SID_ARTIFACT].location,
            "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
        )

    def test_write_then_read_from_disk(self) -> None:
        path = bundle.manifest_path(self.test_dir, "v2")
        bundle.write_manifest(path, _manifest())
        self.assertTrue(os.path.exists(path))
        self.assertEqual(bundle.read_manifest(path).bundle_uuid, "uuid-v2")

    def test_absent_manifest_reports_an_unfinished_run(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "did not finish"):
            bundle.read_manifest(bundle.manifest_path(self.test_dir, "missing"))


if __name__ == "__main__":
    unittest.main()
