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
            bundle.GROUPS_ARTIFACT: bundle.ArtifactEntry(
                save_type="parquet",
                rows=4,
                path="./sid/sid_to_item_groups/v2",
            )
        },
    )
    base.update(kw)
    return bundle.BundleManifest(**base)


class BundleLocationTest(unittest.TestCase):
    def test_file_root_puts_the_artifact_in_a_directory(self) -> None:
        self.assertEqual(
            bundle.artifact_location("./sid", bundle.GROUPS_ARTIFACT, "v2"),
            "./sid/sid_to_item_groups/v2",
        )

    def test_odps_root_makes_the_artifact_a_partition_key(self) -> None:
        self.assertEqual(
            bundle.artifact_location(
                "odps://prj/tables/sid", bundle.MAP_ARTIFACT, "v2"
            ),
            "odps://prj/tables/sid/artifact=item_to_sid_map/generation=v2",
        )

    def test_odps_root_tolerates_a_trailing_slash(self) -> None:
        self.assertEqual(
            bundle.artifact_location(
                "odps://prj/tables/sid/", bundle.MAP_ARTIFACT, "v1"
            ),
            "odps://prj/tables/sid/artifact=item_to_sid_map/generation=v1",
        )

    def test_file_read_pattern_globs_the_part_files(self) -> None:
        self.assertEqual(
            bundle.artifact_read_pattern("./sid", bundle.MAP_ARTIFACT, "v1", "parquet"),
            "./sid/item_to_sid_map/v1/part-*.parquet",
        )

    def test_odps_read_pattern_is_the_partition_itself(self) -> None:
        self.assertEqual(
            bundle.artifact_read_pattern(
                "odps://prj/tables/sid", bundle.MAP_ARTIFACT, "v1", "parquet"
            ),
            "odps://prj/tables/sid/artifact=item_to_sid_map/generation=v1",
        )

    def test_manifest_path_is_per_generation(self) -> None:
        self.assertEqual(bundle.manifest_path("./sid", "v2"), "./sid/manifest/v2.json")


class BundleLayoutTest(unittest.TestCase):
    def _layout(self, **kw):
        base = dict(
            map_root="./map",
            bundle_root="./sid",
            partition="v2",
            from_partition="v1",
            reader_type=None,
        )
        base.update(kw)
        return bundle.resolve_layout(**base)

    def test_each_family_resolves_under_its_own_root(self) -> None:
        layout = self._layout()
        self.assertEqual(
            layout.write_path(bundle.MAP_ARTIFACT), "./map/item_to_sid_map/v2"
        )
        self.assertEqual(
            layout.write_path(bundle.DELTA_MAP_ARTIFACT),
            "./map/delta_item_to_sid_map/v2",
        )
        self.assertEqual(
            layout.write_path(bundle.GROUPS_ARTIFACT), "./sid/sid_to_item_groups/v2"
        )
        self.assertEqual(
            layout.write_path(bundle.DELTA_GROUPS_ARTIFACT),
            "./sid/delta_sid_to_item_groups/v2",
        )

    def test_reads_come_from_the_appended_onto_partition(self) -> None:
        layout = self._layout()
        self.assertEqual(
            layout.read_path(bundle.MAP_ARTIFACT),
            "./map/item_to_sid_map/v1/part-*.parquet",
        )
        self.assertEqual(
            layout.read_path(bundle.GROUPS_ARTIFACT),
            "./sid/sid_to_item_groups/v1/part-*.parquet",
        )

    def test_reader_type_selects_the_map_suffix_but_never_the_groups(self) -> None:
        layout = self._layout(reader_type="CsvReader")
        self.assertTrue(layout.read_path(bundle.MAP_ARTIFACT).endswith("part-*.csv"))
        self.assertTrue(
            layout.read_path(bundle.GROUPS_ARTIFACT).endswith("part-*.parquet")
        )

    def test_an_odps_map_root_leaves_the_serving_set_on_files(self) -> None:
        layout = self._layout(map_root="odps://prj/tables/sid")
        self.assertEqual(
            layout.write_path(bundle.MAP_ARTIFACT),
            "odps://prj/tables/sid/artifact=item_to_sid_map/generation=v2",
        )
        self.assertEqual(
            layout.write_path(bundle.GROUPS_ARTIFACT), "./sid/sid_to_item_groups/v2"
        )

    def test_manifest_locations_track_the_two_partitions(self) -> None:
        layout = self._layout()
        self.assertEqual(layout.prior_manifest_path(), "./sid/manifest/v1.json")

    def test_a_missing_root_is_reported_against_its_artifact(self) -> None:
        layout = self._layout(map_root=None)
        with self.assertRaisesRegex(RuntimeError, "item_to_sid_map"):
            layout.write_path(bundle.MAP_ARTIFACT)
        self.assertEqual(
            layout.write_path(bundle.GROUPS_ARTIFACT), "./sid/sid_to_item_groups/v2"
        )

    def test_a_full_resolve_has_nothing_to_read(self) -> None:
        layout = self._layout(from_partition=None)
        with self.assertRaisesRegex(RuntimeError, "from_partition"):
            layout.read_path(bundle.GROUPS_ARTIFACT)


class BundleManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def test_round_trip_preserves_every_field(self) -> None:
        original = _manifest(
            since_bundle_uuid="uuid-v1", source_model="rqvae", item_id_type="int64"
        )
        restored = bundle.BundleManifest.from_json(original.to_json())
        self.assertEqual(restored, original)

    def test_absent_optional_fields_are_omitted_not_null(self) -> None:
        payload = json.loads(_manifest().to_json())
        self.assertNotIn("since_bundle_uuid", payload)
        self.assertNotIn("source_model", payload)
        entry = payload["artifacts"][bundle.GROUPS_ARTIFACT]
        self.assertNotIn("table", entry)
        self.assertIn("path", entry)

    def test_rejects_a_manifest_missing_required_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            bundle.BundleManifest.from_json(json.dumps({"bundle_uuid": "x"}))

    def test_rejects_a_manifest_without_the_observed_ceiling(self) -> None:
        payload = json.loads(_manifest().to_json())
        del payload["max_observed_items_per_sid"]
        with self.assertRaisesRegex(ValueError, "max_observed_items_per_sid"):
            bundle.BundleManifest.from_json(json.dumps(payload))

    def test_artifact_entry_records_an_odps_table_and_partition(self) -> None:
        entry = bundle.artifact_entry(
            "odps://prj/tables/sid", bundle.MAP_ARTIFACT, "v2", "parquet", 9
        )
        self.assertEqual(entry.save_type, "odps")
        self.assertEqual(entry.table, "sid")
        self.assertEqual(entry.partition, "artifact=item_to_sid_map/generation=v2")
        self.assertIsNone(entry.path)

    def test_artifact_entry_records_a_file_path(self) -> None:
        entry = bundle.artifact_entry(
            "./sid", bundle.GROUPS_ARTIFACT, "v2", "parquet", 4
        )
        self.assertEqual(entry.save_type, "parquet")
        self.assertEqual(entry.path, "./sid/sid_to_item_groups/v2")
        self.assertIsNone(entry.table)

    def test_rejects_a_serving_artifact_that_is_not_parquet(self) -> None:
        payload = json.loads(_manifest().to_json())
        payload["artifacts"][bundle.GROUPS_ARTIFACT]["save_type"] = "odps"
        with self.assertRaisesRegex(ValueError, "serving set is always parquet"):
            bundle.BundleManifest.from_json(json.dumps(payload))

    def test_map_artifact_may_be_odps(self) -> None:
        payload = json.loads(_manifest().to_json())
        payload["artifacts"][bundle.MAP_ARTIFACT] = {
            "save_type": "odps",
            "rows": 9,
            "table": "prj.sid",
            "partition": "artifact=item_to_sid_map/generation=v2",
        }
        restored = bundle.BundleManifest.from_json(json.dumps(payload))
        self.assertEqual(restored.artifacts[bundle.MAP_ARTIFACT].table, "prj.sid")

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
