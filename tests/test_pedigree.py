import sys
from pathlib import Path
import pytest
import numpy as np
import itertools as it

sys.path.insert(0, str(Path(__file__).parent.parent))

from ponderosa.pedigree import PedigreeHierarchy, PedigreeRegistry, Siblings, Relationship, PedigreeCodes, PedigreeGraph, RelationshipPathFinder

FIRST_SECOND_REL_DICT = {
    'FS': {1: [[1, -1]], 2: [[1, -1]], 'sex': False, 'degree': '1st'},
    'PO': {1: [[1]], 'sex': False, 'degree': '1st'},
    'HS': {1: [[1, -1]], 'sex': False, 'degree': '2nd'},
    'AV': {1: [[1, 1, -1], [1, 1, -1]], 'sex': False, 'degree': '2nd', "parent": "GPAV"},
    'GP': {1: [[1, 1]], 'sex': False, 'degree': '2nd', "parent": "GPAV"},
    'MGP': {2: [[1, 1]], 'sex': True, 'degree': '2nd', 'parent': 'GP'},
    'PGP': {1: [[1, 1]], 'sex': True, 'degree': '2nd', 'parent': 'GP'},
    'MHS': {2: [[1, -1]], 'sex': True, 'degree': '2nd', 'parent': 'HS'},
    'PHS': {1: [[1, -1]], 'sex': True, 'degree': '2nd', 'parent': 'HS'},
    "GPAV": {"sex": False, "degree": "2nd"}
}


class TestPedigreeHierarchy:

    hierarchy = PedigreeHierarchy(FIRST_SECOND_REL_DICT)

    def test_basic_second(self):

        assert ("relatives", "2nd") in self.hierarchy.edges
        assert ("relatives", "1st") in self.hierarchy.edges

        for nodes in self.hierarchy.nodes:
            if nodes in ["relatives", "2nd", "1st"]:
                continue
            else:
                assert ("relatives", nodes) not in self.hierarchy.edges


class TestPedigreeRegistry:

    def _create_basic_second_registry(self):

        return PedigreeRegistry.from_dict(FIRST_SECOND_REL_DICT)

    def test_basic_second_cache(self):

        registry = self._create_basic_second_registry()

        _ = registry.get_relatives("HS")

        assert "HS" in registry._cache
        assert "HS" in registry._cache["HS"]
        assert "PHS" in registry._cache["HS"]
        assert "MHS" in registry._cache["HS"]

        _ = registry.get_relatives("PHS")

        assert "PHS" in registry._cache

    def test_basic_second(self):
        # TODO test new GPAV node

        registry = self._create_basic_second_registry()

        registry.add_pair("A", "B", "PHS")

        test1 = np.array([["A", "B", "PHS"]])

        # assertions
        np.testing.assert_array_equal(test1, registry.get_relatives_from(["PHS"]))
        np.testing.assert_array_equal(test1, registry.get_relatives_from(["PHS"]))
        np.testing.assert_array_equal(test1, registry.get_relatives_from(["PHS", "MHS"]))

        test2 = np.array([["A", "B", "PHS"],
                          ["A", "C", "MHS"]])

        registry.add_pair("A", "C", "MHS")

        np.testing.assert_array_equal(test2, registry.get_relatives_from(["PHS", "MHS"]))

        test3 = np.array([["A", "B"]])

        np.testing.assert_array_equal(test3, registry.get_relatives("PHS"))

        registry.add_pair("D", "A", "AV")

        test4 = np.array([["A", "D"]])
        test5 = np.array([["D", "A"]])

        np.testing.assert_array_equal(test4, registry.get_relatives("AV", ordered=True))
        np.testing.assert_array_equal(test5, registry.get_relatives("AV", ordered=False))



class TestRelationship:

    testing_dict = {
        "PHS": np.array([[1, 1, -1]]),
        "MHS": np.array([[2, 1, -1]]),
        "PAV": np.array([[1, 1, 1, -1],
                        [1, 1, 1, -1]]),
        "MAV": np.array([[2, 1, 1, -1],
                        [2, 1, 1, -1]]),
        "MGP": np.array([[2, 1, 1]]),
        "PGP": np.array([[1, 1, 1]]),
        "FATHER": np.array([[1, 1]]),
        "MOTHER": np.array([[2, 1]]),
    }

    def _get_matrices(self, as_true: list):

        return_dict = {}
        for i, j in self.testing_dict.items():
            return_dict[i] = [j, i in as_true]

        return return_dict.items()

    def _test_relationship_matches(self, relationship_key, expected_matches):
        """Test that a relationship correctly identifies expected matches."""
        rel_dict = FIRST_SECOND_REL_DICT[relationship_key]
        R = Relationship(rel_dict)
        
        for rel_name, (mat, expected) in self._get_matrices(expected_matches):
            assert R.is_relationship(mat) == expected, f"Failed for {rel_name} with {relationship_key}"

    @pytest.mark.parametrize("rel,paternal,maternal", [
        pytest.param("GP", "PGP", "MGP", id="gp"),
        pytest.param("HS", "PHS", "MHS", id="hs"), 
        ])
    def test_sex_specific(self, rel, paternal, maternal):

        # Test non-sex-specific (should match both)
        self._test_relationship_matches(rel, [paternal, maternal])
        
        # Test each sex-specific version (should match only itself)
        self._test_relationship_matches(paternal, [paternal])
        self._test_relationship_matches(maternal, [maternal])

    def test_typeIII(self):

        rel_dict = {
            1: [[1, 1, -1, -1]],
            2: [[1, 1, -1, -1]],
            "sex": False
        }

        R = Relationship(rel_dict)

        mat = [[1, 1, 1, -1, -1],
                [2, 1, 1, -1, -1]]

        assert R.is_relationship(sorted(mat, reverse=True))
        assert R.is_relationship(sorted(mat, reverse=False))



class TestPedigreeCodes:

    def test_basic(self):

        codes = PedigreeCodes(FIRST_SECOND_REL_DICT)

        paths = {
            1: [[1, 1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "PGP"
        assert ibd1 == 0.50
        assert ibd2 == 0.0
        assert same_gen == False

        paths = {
            1: [[1, 1, -1], [1, 1, -1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "AV"
        assert ibd1 == 0.50
        assert ibd2 == 0.0
        assert same_gen == False

        paths = {
            1: [[1, 1, -1]],
            2: [[1, 1, -1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "unknown"
        assert ibd1 == (0.25*0.75*2)
        assert ibd2 == (0.25 * 0.25)

        paths = {
            1: [[1, -1]],
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "PHS"
        assert same_gen
        assert ibd1 == 0.50
        assert ibd2 == 0

        paths = {
            1: [[1, -1, -1, -1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "nan"
        assert ibd1 == 0.125
        assert ibd2 == 0

        paths = {
            1: [[-1, 1, 1, 1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "unknown"
        assert ibd1 == 0.125
        assert ibd2 == 0

        paths = {
            1: [[1, -1, -1], [1, -1, -1]]
        }

        rel, ibd1, ibd2, _, same_gen = codes.determine_relationship(paths)

        assert rel == "nan"
        assert ibd1 == 0.50
        assert ibd2 == 0


class TestPedigree:

    def test_basic(self):

        po_list = [["A", "B"],
                   ["A", "C"],
                   ["D", "E"],
                   ["0", "B"],
                   ["0", "C"],
                   ["0", "E"]
        ]

        sex_list = [["A", 1],
                    ["B", 1],
                    ["C", 1],
                    ["D", 2],
                    ["E", 2]
        ]

        default_missing_parent = ["0"]

        pedigree = PedigreeGraph(po_list, default_missing_parent, sex_list)

        assert pedigree.has_mother("A") == False
        assert pedigree.has_mother("B") == False
        assert pedigree.has_mother("D") == False
        assert pedigree.has_mother("E") == True
        assert pedigree.has_father("E") == False
        assert pedigree.has_father("B") == True

        pedigree.add_parents_from(
            [
                ["Dummy1", "B", {"Dummy1": 1}],
                ["Dummy1", "C", {"Dummy1": 1}],
                ["Dummy2", "B", {"Dummy2": 2}],
                ["Dummy2", "C", {"Dummy2": 2}]
            ]
        )

        assert pedigree.has_mother("B") == True
        assert pedigree.has_mother("C") == True
        assert "Dummy1" not in pedigree._get_parents("B")
        assert "Dummy1" not in pedigree._get_parents("C")
        assert "Dummy2" in pedigree._get_parents("B")
        assert "Dummy2" in pedigree._get_parents("C")


class TestRelationshipPathFinder:

    def test_basic(self):

        po_list = [["A", "C"],
                    ["B", "C"],
                    ["A", "D"],
                    ["B", "D"],
                    ["D", "E"],
                    ["A", "F"],
                    ["G", "F"],
                    ["E", "H"]
                ]

        sex_list = [["A", 1],
                    ["B", 2],
                    ["C", 1],
                    ["D", 2],
                    ["E", 1],
                    ["F", 1],
                    ["G", 2],
                    ["H", 1]
        ]

        default_missing_parent = ["0"]

        path_finder = RelationshipPathFinder.from_edge_list(
            po_list,
            default_missing_parent,
            sex_list
        )

        codes = PedigreeCodes(FIRST_SECOND_REL_DICT)

        confirmed = {
            ("C", "A"): "PO",
            ("C", "B"): "PO",
            ("D", "A"): "PO",
            ("D", "B"): "PO",
            ("E", "D"): "PO",
            ("E", "C"): "AV",
            ("F", "G"): "PO",
            ("F", "A"): "PO",
            ("C", "F"): "PHS",
            ("D", "F"): "PHS",
            ("E", "A"): "MGP",
            ("E", "B"): "MGP",
            ("C", "D"): "FS",
            ("H", "E"): "PO",
            ("H", "D"): "PGP"
        }

        n_confirmed = 0

        for focal,_ in sex_list:

            pairs = path_finder.find_focal_relationships(focal, codes)

            for id1, id2, rname in pairs:
                assert id1 == focal
                assert (id1, id2) in confirmed
                assert confirmed[(id1, id2)] == rname
                n_confirmed += 1

        assert n_confirmed == len(confirmed)

        assert len(codes.unknown_codes) == 4

    def test_full_basic(self):
        po_list = [["A", "C"],
                    ["B", "C"],
                    ["A", "D"],
                    ["B", "D"],
                    ["D", "E"],
                    ["A", "F"],
                    ["G", "F"],
                    ["E", "H"],
                    ["B", "J"],
                    ["I", "J"],
                    ["C", "K"]
                ]

        sex_list = [["A", 1],
                    ["B", 2],
                    ["C", 1],
                    ["D", 2],
                    ["E", 1],
                    ["F", 1],
                    ["G", 2],
                    ["H", 1],
                    ["I", 1],
                    ["J", 2],
                    ["K", 2]
        ]

        default_missing_parent = ["0"]

        path_finder = RelationshipPathFinder.from_edge_list(
            po_list,
            default_missing_parent,
            sex_list
        )

        codes = PedigreeCodes(FIRST_SECOND_REL_DICT)

        registry = PedigreeRegistry.from_dict(FIRST_SECOND_REL_DICT)

        path_finder.find_all_relationships(registry, codes)

        assert registry.get_relatives("HS").shape[0] == 4
        assert registry.get_relatives("PHS").shape[0] == 2
        assert registry.get_relatives("MHS").shape[0] == 2
        assert registry.get_relatives("AV").shape[0] == 2
        assert registry.get_relatives("GP").shape[0] == 5
        assert registry.get_relatives("MGP").shape[0] == 2
        assert registry.get_relatives("PGP").shape[0] == 3
        assert registry.get_relatives("2nd").shape[0] == 11


class TestSiblings:

    def _is_relationship(self, siblings: Siblings, id1: str, id2: str, rel: str) -> bool:
        pair1, pair2 = (id1, id2), (id2, id1)
        if rel == "HS":
            if pair1 in siblings.HS or pair2 in siblings.HS:
                return True
            if pair1 in siblings.predHS or pair2 in siblings.predHS:
                return True
        if rel == "FS":
            if pair1 in siblings.FS or pair2 in siblings.FS:
                return True
            if pair1 in siblings.predFS or pair2 in siblings.predFS:
                return True
        return False

    def _generate_test_data(self, n_fs=2, n_hs=2, n_unresolved=2):
        """Generate realistic test data for sibling classification"""
        pairs = []
        labels = []
        true_labels = []
        ibd_data = []
        
        # Generate FS pairs (higher IBD2)
        for i in range(n_fs):
            pairs.append((f"FS{i}_A", f"FS{i}_B"))
            labels.append("FS"); true_labels.append("FS")
            ibd_data.append([0.5, np.random.normal(0.25, 0.05)])  # IBD2 ~0.25
        
        # Generate HS pairs (low/no IBD2)
        for i in range(n_hs):
            pairs.append((f"HS{i}_A", f"HS{i}_B"))
            labels.append("HS"); true_labels.append("HS")
            ibd_data.append([0.5, np.random.normal(0.0, 0.01)])   # IBD2 ~0
        
        # Generate unresolved pairs
        for i in range(n_unresolved):
            pairs.append((f"U{i}_A", f"U{i}_B"))
            labels.append("unresolved")
            if np.random.choice([0, 1]):
                ibd_data.append([0.5, np.random.normal(0.0, 0.01)]) 
                true_labels.append("HS")
            else:
                ibd_data.append([0.5, np.random.normal(0.25, 0.05)])
                true_labels.append("FS")
            
        return np.array(true_labels), np.array(pairs), np.array(labels), np.abs(np.array(ibd_data))

    @pytest.mark.parametrize("n_fs,n_hs,n_unresolved", [
    pytest.param(0, 0, 5, id="no_training"),
    pytest.param(10, 10, 5, id="balanced"), 
    pytest.param(10, 10, 0, id="no_unresolved"),
    pytest.param(0, 0, 0, id="no_samples")
    ])
    def test_basic_classification(self, n_fs: int, n_hs: int, n_unresolved: int):

        true_labels, pairs, labels, ibd_data = self._generate_test_data(n_fs, n_hs, n_unresolved)
        
        siblings = Siblings(pairs, labels, ibd_data)

        for pair, truth in zip(pairs, true_labels):

            assert self._is_relationship(siblings, *pair, truth)

    def test_dummy_parents(self):

        def iter_set(sibling_set):
            sibling_set = it.chain(*sibling_set)
            return it.combinations(sibling_set, r=2)

        pairs = np.array([("A", "B"),
                 ("A", "C"),
                 ("B", "C"),
                 ("D", "E"),
                 ("E", "F"),
                 ("G", "H"),
                 ("I", "J")])

        labels = np.array(["unresolved"] * pairs.shape[0])

        ibd_data = np.array([
                            [0.5, 0.25],
                            [0.5, 0.25],
                            [0.5, 0.25],
                            [0.5, 0.25],
                            [0.5, 0.25],
                            [0.5, 0],
                            [0.5, 0]
        ])

        siblings = Siblings(pairs, labels, ibd_data)

        parent_dict = {}

        for parent, child, data in siblings.get_dummy_parent_edges():

            if child not in parent_dict:
                parent_dict[child] = [0, 0]

            parent_dict[child][data[parent]["SEX"]-1] = parent

        set1 = pairs[:3]
        set2 = pairs[3:5]

        for fs_set in [set1, set2]:
            for id1, id2 in iter_set(fs_set):
                assert parent_dict[id1][0] == parent_dict[id2][0]
                assert parent_dict[id1][1] == parent_dict[id2][1]
                assert parent_dict[id1][0] != 0
                assert parent_dict[id1][1] != 0