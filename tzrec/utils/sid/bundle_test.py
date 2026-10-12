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

from parameterized import parameterized

from tzrec.utils.sid import bundle
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


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


class BundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def _bundle(self, **kw):
        base = dict(
            output_path="./sid",
            item_to_sid_root="./map",
            generation="v2",
            from_generation="v1",
        )
        base.update(kw)
        return bundle.Bundle(**base)

    def test_each_family_resolves_under_its_own_root(self) -> None:
        layout = self._bundle()
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

    @parameterized.expand(
        [
            ("file_root", "./map", "CsvWriter", "./map/v1/item_to_sid/part-*.csv"),
            (
                "odps_root",
                "odps://prj/tables/sid",
                "ParquetWriter",
                "odps://prj/tables/sid/generation=v1/artifact=item_to_sid",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_reads_come_from_the_appended_onto_generation(
        self, _name, item_to_sid_root, writer_type, item_to_sid_path
    ) -> None:
        prior = bundle.Bundle(self.test_dir, item_to_sid_root, "v1", None)
        prior.rows[bundle.SID_TO_ITEMS_ARTIFACT] = 2
        prior.rows[bundle.ITEM_TO_SID_ARTIFACT] = 3
        prior.publish([8, 8], 5, 3, "int64", writer_type)
        layout = self._bundle(
            output_path=self.test_dir, item_to_sid_root=item_to_sid_root
        )
        self.assertEqual(
            layout.read_path(bundle.ITEM_TO_SID_ARTIFACT), item_to_sid_path
        )
        self.assertEqual(
            layout.read_path(bundle.SID_TO_ITEMS_ARTIFACT),
            os.path.join(self.test_dir, "v1", "sid_to_items", "part-*.parquet"),
        )

    def test_an_odps_item_to_sid_root_leaves_the_serving_set_on_files(self) -> None:
        layout = self._bundle(item_to_sid_root="odps://prj/tables/sid")
        self.assertEqual(
            layout.write_path(bundle.ITEM_TO_SID_ARTIFACT),
            "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
        )
        self.assertEqual(
            layout.write_path(bundle.SID_TO_ITEMS_ARTIFACT), "./sid/v2/sid_to_items"
        )

    def test_a_missing_root_is_reported_against_its_artifact(self) -> None:
        layout = self._bundle(item_to_sid_root=None)
        with self.assertRaisesRegex(RuntimeError, "item_to_sid"):
            layout.write_path(bundle.ITEM_TO_SID_ARTIFACT)
        self.assertEqual(
            layout.write_path(bundle.SID_TO_ITEMS_ARTIFACT), "./sid/v2/sid_to_items"
        )

    def test_an_odps_output_path_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "always files"):
            self._bundle(output_path="odps://prj/tables/g", from_generation=None)

    def test_a_full_resolve_has_nothing_to_read(self) -> None:
        layout = self._bundle(from_generation=None)
        with self.assertRaisesRegex(RuntimeError, "from_generation"):
            layout.read_path(bundle.SID_TO_ITEMS_ARTIFACT)

    @parameterized.expand(
        [
            (
                "odps_item_to_sid",
                "odps://prj/tables/sid",
                bundle.ITEM_TO_SID_ARTIFACT,
                "ParquetWriter",
                "odps",
                "odps://prj/tables/sid/generation=v2/artifact=item_to_sid",
            ),
            (
                "file_item_to_sid",
                "./map",
                bundle.ITEM_TO_SID_ARTIFACT,
                "CsvWriter",
                "csv",
                "./map/v2/item_to_sid",
            ),
            (
                "serving_set_under_a_csv_writer",
                "./map",
                bundle.SID_TO_ITEMS_ARTIFACT,
                "CsvWriter",
                "parquet",
                "{out}/v2/sid_to_items",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_publish_records_where_and_how_each_artifact_landed(
        self, _name, item_to_sid_root, artifact, writer_type, save_type, location
    ) -> None:
        published = self._bundle(
            output_path=self.test_dir,
            item_to_sid_root=item_to_sid_root,
            from_generation=None,
        )
        published.rows[artifact] = 4
        path = published.publish([8, 8], 5, 3, "int64", writer_type)
        entry = bundle.read_manifest(path).artifacts[artifact]
        self.assertEqual(entry.save_type, save_type)
        self.assertEqual(entry.location, location.format(out=self.test_dir))


class BundleManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def test_publish_round_trips_every_field_through_disk(self) -> None:
        prior = bundle.Bundle(self.test_dir, "./map", "v1", None)
        prior.rows[bundle.SID_TO_ITEMS_ARTIFACT] = 1
        prior.publish([8, 8], 5, 7, "int64", "ParquetWriter")
        appended = bundle.Bundle(self.test_dir, "./map", "v2", "v1")
        appended.rows[bundle.SID_TO_ITEMS_ARTIFACT] = 4

        path = appended.publish([8, 8], 5, 3, "int64", "ParquetWriter")

        self.assertEqual(path, bundle.manifest_path(self.test_dir, "v2"))
        self.assertEqual(
            bundle.read_manifest(path),
            _manifest(
                bundle_uuid=appended.uuid,
                since_bundle_uuid=prior.uuid,
                max_observed_items_per_sid=7,
                item_id_type="int64",
                artifacts={
                    bundle.SID_TO_ITEMS_ARTIFACT: bundle.ArtifactEntry(
                        save_type="parquet",
                        rows=4,
                        location=os.path.join(self.test_dir, "v2", "sid_to_items"),
                    )
                },
            ),
        )

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

    def test_absent_manifest_reports_an_unfinished_run(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "did not finish"):
            bundle.read_manifest(bundle.manifest_path(self.test_dir, "missing"))


if __name__ == "__main__":
    unittest.main()
