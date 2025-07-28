import numpy as np
import pandas as pd

def simple_ibd_segments(kinship: float, id1: str, id2: str, id1_haplotype: list, id2_haplotype: list, n_chrom: int = 2):

    segments = []

    for c in range(1, n_chrom + 1):

        n_segments = max(1, np.random.poisson(4))

        if kinship == 1:
            boundaries = np.linspace(0, 100, n_segments + 1)

            for i in range(n_segments):
                start, end = boundaries[[i,i+1]]
                segments.append([id1, np.random.choice(id1_haplotype),
                                 id2, np.random.choice(id2_haplotype),
                                 c, start, end]) 

            continue

        actual_cm = np.random.normal(kinship, kinship / 5) * 100

        seg_len = actual_cm / n_segments

        boundaries = np.linspace(0, 100, n_segments + 1)

        for i in range(n_segments):
            start, end = boundaries[[i,i+1]]

            start = np.random.uniform(start, end - seg_len)
            end = start + seg_len

            segment = [id1, np.random.choice(id1_haplotype),
                       id2, np.random.choice(id2_haplotype),
                       c, start, end]

            segments.append(segment)

    return np.array(segments, dtype=object)

            

def generate_simple_segments_for(rel_type, id1, id2):
    if rel_type == "PO":
        segments = simple_ibd_segments(1, id1, id2, np.random.choice([0, 1], 1), [0, 1])

    elif rel_type == "FS":
        segments1 = simple_ibd_segments(0.5, id1, id2, [0], [0])
        segments2 = simple_ibd_segments(0.5, id1, id2, [1], [1])
        segments = np.vstack((segments1, segments2))

    elif rel_type in ["HS", "AV", "GP", "MGP", "PGP", "MHS", "PHS", "GPAV"]:
        segments = simple_ibd_segments(0.5, id1, id2, np.random.choice([0, 1], 1), [0, 1])

    elif rel_type in ["CO"]:
        segments = simple_ibd_segments(0.25, id1, id2, np.random.choice([0, 1], 1), [0, 1])

    elif rel_type in ["HCO"]:
        segments = simple_ibd_segments(0.125, id1, id2, np.random.choice([0, 1], 1), [0, 1])

    elif rel_type in ["HSCO"]:
        segments1 = simple_ibd_segments(0.5, id1, id2, [0], [0])
        segments2 = simple_ibd_segments(0.25, id1, id2, [1], [1])
        segments = np.vstack((segments1, segments2))

    return segments


class GeneratePairs:

    def __init__(self, n_pairs: int = 10):

        self.n_pairs = n_pairs
        self.id_index = 1
        self.segments = []
        self.ages = []
        self.fam = []

        self.generation_ages = {
            1: [0, 10],
            2: [30, 50],
            3: [70, 90]
        }

    def _new_pair(self):
        self.id_index += 2
        return f"ID{self.id_index-2}", f"ID{self.id_index-1}"

    def _new_ind(self):
        self.id_index += 1
        return f"ID{self.id_index-1}"

    def _sex(self):
        return np.random.choice([1, 2])

    def _age(self, gen):
        return np.random.randint(*self.generation_ages[gen])

    def add_fam_line(self, iid, sex, father, mother, gen = None):
        self.fam.append(["0", iid, father, mother, sex, "-9"])

        if gen:
            self.ages.append([iid, self._age(gen)])

    def add_segments(self, rel_type, id1, id2):
        segment_arr = generate_simple_segments_for(rel_type, id1, id2)
        self.segments.append(segment_arr)

    def po(self):
        for _ in range(self.n_pairs):
            id1, id2 = self._new_pair()
            self.add_segments("PO", id1, id2)

            parent_sex = self._sex()

            father = id2 if parent_sex == 1 else self._new_ind()
            mother = id2 if parent_sex == 2 else self._new_ind()

            gen = np.random.choice([1, 2])

            self.add_fam_line(id1, self._sex(), father, mother, gen=gen)
            self.add_fam_line(father, 1, "0", "0", gen=gen+1)
            self.add_fam_line(mother, 2, "0", "0", gen=gen+1)

    def fs(self):
        for _ in range(self.n_pairs):
            id1, id2 = self._new_pair()

            mother, father = self._new_pair()

            gen = np.random.choice([1, 2])

            self.add_fam_line(id1, self._sex(), father, mother, gen=gen)
            self.add_fam_line(id2, self._sex(), father, mother, gen=gen)
            self.add_fam_line(father, 1, "0", "0", gen=gen+1)
            self.add_fam_line(mother, 2, "0", "0", gen=gen+1)

            self.add_segments("FS", id1, id2)
            

    def hs(self, parent_sex: int):
        for _ in range(self.n_pairs):
            id1, id2 = self._new_pair()
            self.add_segments("PHS" if parent_sex == 1 else "MHS", id1, id2)

            gen = np.random.choice([1, 2])

            shared_parent = self._new_ind()
            self.add_fam_line(shared_parent, parent_sex, "0", "0", gen=gen+1)

            for iid in [id1, id2]:
                father = shared_parent if parent_sex == 1 else self._new_ind()
                mother = shared_parent if parent_sex == 2 else self._new_ind()
                self.add_fam_line(iid, self._sex(), father, mother, gen=gen)
                self.add_fam_line(mother if father == shared_parent else father, {1:2, 2:1}[parent_sex], "0", "0", gen=gen+1)

    def av(self):
        for _ in range(self.n_pairs):
            parent, uncle = self._new_pair()
            gma, gpa = self._new_pair()
            niece = self._new_ind()

            self.add_segments("AV", niece, uncle)

            self.add_fam_line(niece, self._sex(), parent, "0", gen=1)
            self.add_fam_line(parent, 1, gpa, gma, gen=2)
            self.add_fam_line(uncle, self._sex(), gpa, gma, gen=2)
            self.add_fam_line(gpa, 1, "0", "0", gen=3)
            self.add_fam_line(gma, 2, "0", "0", gen=3)

    def gp(self, parent_sex: int):
        for _ in range(self.n_pairs):
            id1 = self._new_ind()
            id2 = self._new_ind()
            id3 = self._new_ind()

            self.add_segments("PGP" if parent_sex == 1 else "MGP", id1, id3)

            father = id2 if parent_sex == 1 else self._new_ind()
            mother = id2 if parent_sex == 2 else self._new_ind()

            self.add_fam_line(id1, self._sex(), father, mother, gen=1)

            father = id3 if self._sex() == 1 else self._new_ind()
            mother = self._new_ind() if father == id3 else id3

            self.add_fam_line(id2, parent_sex, father, mother, gen=2)
            self.add_fam_line(father, 1, "0", "0", gen=3)
            self.add_fam_line(mother, 2, "0", "0", gen=3)

    def co(self, half: bool = False):

        co1, co2 = self._new_pair()

        self.add_segments("HCO" if half else "CO", co1, co2)

        sibs = []
        for co in [co1, co2]:
            father, mother = self._new_pair()
            sibs.append(mother)
            self.add_fam_line(co, self._sex(), father, mother, gen=1)
            self.add_fam_line(father, 1, "0", "0", gen=2)

        if half:
            shared_mother = self._new_ind()
            self.add_fam_line(shared_mother, 2, "0", "0", gen=3)
            for sib in sibs:
                father = self._new_ind()
                self.add_fam_line(sib, self._sex(), father, shared_mother, gen=2)
                self.add_fam_line(father, 1, "0", "0", gen=3)
        else:
            father, mother = self._new_pair()
            self.add_fam_line(father, 1, "0", "0", gen=3)
            self.add_fam_line(mother, 2, "0", "0", gen=3)
            for sib in sibs:
                self.add_fam_line(sib, self._sex(), father, mother, gen=2)

    def hsco(self):

        co1, co2 = self._new_pair()

        self.add_segments("HSCO", co1, co2)

        mother = self._new_ind()
        father1 = self._new_ind()
        father2 = self._new_ind()

        gma, gpa = self._new_pair()

        self.add_fam_line(co1, self._sex(), father1, mother, gen=1)
        self.add_fam_line(co2, self._sex(), father2, mother, gen=1)
        self.add_fam_line(mother, 2, "0", "0", gen=2)
        for father in [father1, father2]:
            self.add_fam_line(father, 1, gpa, gma, gen=2)

        self.add_fam_line(gpa, 1, "0", "0", gen=3)
        self.add_fam_line(gma, 2, "0", "0", gen=3)


    def generate_all(self):
        self.po()
        self.fs()
        self.hs(1)
        self.hs(2)
        self.av()
        self.gp(1)
        self.gp(2)
        self.co(half=True)
        self.co(half=False)
        self.hsco()

    def write_out(self, path_and_prefix: str, delim="\t", n_chrom=2):

        segments = np.vstack(self.segments)

        cols = ["id1", "id1_haplotype", "id2", "id2_haplotype", "chromosome", "start_cm", "end_cm"]

        # Save segments
        segment_df = pd.DataFrame(segments, columns=cols)
        segment_df.to_csv(f"{path_and_prefix}_segments.txt", sep=delim, index=False)

        # Save age file
        if len(self.ages) > 0:
            age_df = pd.DataFrame(self.ages)
            age_df.to_csv(f"{path_and_prefix}_ages.txt", sep=delim, index=False, header=False)

        # Save fam file
        fam_df = pd.DataFrame(self.fam)
        fam_df.to_csv(f"{path_and_prefix}.fam", header=False, index=False, sep=" ")

        dfs = []
        for i in range(1, n_chrom + 1):
            map_df = pd.DataFrame(
                {"chr": [i] * 100,
                "cm": np.linspace(0, 100, 100),
                "mb": np.linspace(0, 100_000_000, 100),
                "rsid": [f"c{i}_rs{j}" for j in range(100)]
                }
            )
            dfs.append(map_df)

        pd.concat(dfs)[["chr", "rsid", "cm", "mb"]].to_csv(f"{path_and_prefix}.map", header=False, index=False, sep=" ")


if __name__ == "__main__":

    gen = GeneratePairs(15)
    gen.generate_all()
    gen.write_out("data/test1/test")

    