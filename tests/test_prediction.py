import numpy as np
import pytest 

from ponderosa.prediction import MatrixHierarchy, process_phase_error

NODE_LIST = [["relatives", "1st"],
            ["1st", "PO"], ["1st", "FS"],
            ["relatives", "2nd"],
            ["2nd", "GPAV"], ["2nd", "HS"],
            ["GPAV", "GP"], ["GPAV", "AV"],
            ["GP", "MGP"], ["GP", "PGP"],
            ["HS", "PHS"], ["HS", "MHS"]]

class TestProcessPhaseError:

    def test_basic(self):

        X = np.array([
            [0.5, 0.4, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.6, 0.2]
        ])

        X_out = process_phase_error(X)
        X_exp = np.array([
            [.5/.9, .4/.9],
            [np.nan, np.nan],
            [0.2/0.8, 0.6/0.8]
        ])

        assert np.allclose(X_exp, X_out, equal_nan=True)


class TestMatrixHierarchy:

    METHODS = ["degree", "hap", "nsegs"]

    DEGREE = [np.array([[0.98, 0.02],
                            [0.52, 0.48],
                            [0.02, 0.98],
                            [1, 0],
                            [0, 1]]),
                ["1st", "2nd"],
                "degree"]

    NSEGS = [np.array([ [0.2 for _ in range(5)],
                        [0.2 for _ in range(5)],
                        [0.40, 0, 0, 0.60, 0],
                        [1, 0, 0, 0, 0],
                        [0.2, 0.2, 0.1, 0.2, 0.3]]),
                ["AV", "MGP", "PGP", "MHS", "PHS"],
                "nsegs"]

    HAP = [np.array([ [0.5, 0.25, 0.25],
                      [1, 0, 0],
                      [0.9, 0.05, 0.05],
                      [1, 0, 0],
                      [0.15, 0, 0.85]]),
           np.array(["HS", "GPAV", "Phase error"]),
           "hap"]

    def test_single_level(self):
        mhier = MatrixHierarchy(NODE_LIST, 5, self.METHODS)

        # level 1
        mhier.add_probs(*self.DEGREE)

        mhier.compute_probs()

        pred, prob = mhier.most_probable(0.50)

        exp_pred = np.array(["1st", "1st", "2nd", "1st", "2nd"])
        exp_prob = np.max(self.DEGREE[0],axis=1)

        assert np.array_equal(exp_pred, pred)
        assert np.array_equal(exp_prob, prob)

    def test_two_level(self):

        mhier = MatrixHierarchy(NODE_LIST, 5, self.METHODS)

        # level 1
        mhier.add_probs(*self.DEGREE)
        # level 2
        mhier.add_probs(*self.NSEGS)

        mhier.compute_probs()

        pred, prob = mhier.most_probable(0.5)

        exp_pred = np.array(["1st", "1st", "MHS", "1st", "GPAV"])
        exp_prob = np.array([1, 0.52, 0.98*.6, 1, 0.5])
        assert np.array_equal(exp_pred, pred)

    def test_three_level(self):

        mhier = MatrixHierarchy(NODE_LIST, 5, self.METHODS)

        # level 1
        mhier.add_probs(*self.DEGREE)
        # level 2
        mhier.add_probs(*self.NSEGS)
        # level 3
        X, classes, lab = self.HAP

        X = process_phase_error(X)

        mhier.add_probs(X, classes[:2], "hap")

        mhier.compute_probs()

        # 1st pair: 1st degree
        # 2nd pair: 1st degree
        # 3rd pair: 0.55705 (PHS), 0.92842 (MHS)
        # 4th pair: 1st degree
        # 5th pair: 1 (2nd),

        pred, prob = mhier.most_probable(0.51)

        exp_prob = np.array([0.98, 0.52, 0.9/0.95*.98, 1, 1])
        exp_pred = np.array(["1st", "1st", "MHS", "1st", "2nd"])

        # import pytest; pytest.set_trace()

        assert np.allclose(exp_prob, prob, atol=1e-4)
        assert np.array_equal(exp_pred, pred)

        pred, prob = mhier.most_probable(0.28)

        exp_prob = np.array([0.98, 0.52, 0.9/0.95*.98, 1, 0.3])
        exp_pred = np.array(["1st", "1st", "MHS", "1st", "GP"])

        assert np.allclose(exp_prob, prob, atol=1e-4)
        assert np.array_equal(exp_pred, pred)

        import pytest; pytest.set_trace()



