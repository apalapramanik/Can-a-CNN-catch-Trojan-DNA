# Data

The generated datasets are not stored in this repository. They take about 845 MB and you can recreate them.

## Where the data comes from

- **Raw sequences.** The authors of the original paper collected 716 *E. coli* plasmid DNAs from the Addgene repository (https://www.addgene.org/) with the scripts in `addgene_data_collection/` of their repository. The collected sequences are in their file `addgene_data_collection/all_plasmid_dna.txt`.
- **Reads.** We cut the sequences into non overlapping 1,000 base reads. This gives 4,356 usable reads, the same count as the paper.
- **Payloads.** Fake addresses of the form `name.name:port`. The authors' own example payloads are at https://bitbucket.org/sibleeislam/bio-cyber-hacking.

The authors' repository: https://github.com/sibleeislam/trojan-malware-in-bio-cyber-attacks

## How to recreate the datasets

1. Clone the authors' repository into `authors_repo/` at the root of this project:

   ```bash
   git clone https://github.com/sibleeislam/trojan-malware-in-bio-cyber-attacks authors_repo
   ```

2. Run the generator. The arguments are comma separated lists of fragment sizes, retentions, encryption keys and dataset numbers:

   ```bash
   python3 src/ap_step4d_generate_datasets.py 1,5 1,5 0 0,1,2,3,4,5,6,7,8,9
   ```

   Without arguments it writes the baseline set: retention 0, key 0, fragment sizes 1 to 5, ten datasets.

## Layout

```
data/datasets/fragment_len_{1..5}/retention_pos_{0..5}/encryption_key_{0,10,...}/dataset_{0..9}/
    non_trojan_dataset.txt                        1,000 clean reads
    nw_best_greedy_trojan_insertion_dataset.txt   same hosts, MaxNW placement
    nw_worst_greedy_trojan_insertion_dataset.txt  same hosts, MinNW placement
    random_trojan_insertion_dataset.txt           same hosts, random placement
    payloads.txt, decodes.csv                     payloads and the host id of each line
```

The same 1,000 host reads appear in all ten datasets, as in the paper.
