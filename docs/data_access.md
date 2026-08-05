# Data access and downloads

## Alzheimer's disease GWAS loci

The 38 lead variants come from Wightman et al. The article and supplementary information are at
<https://doi.org/10.1038/s41588-021-00921-z>.

## 1000 Genomes high-coverage panel

The analysis uses the 1000 Genomes 30x high-coverage release aligned to GRCh38. Collection metadata:
<https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x/>.

The chromosome-specific phased SNV/indel/SV VCF release used by this work is under:
<https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/>.

The paper used 503 mutually unrelated EUR individuals (1,006 phased haplotypes) for allele
frequencies, signed phased LD, haplotypes, and fitting. Users should record exact source filenames
and checksums when downloading because upstream collection layouts can evolve.

## ROS/MAP genotype and expression data

The ROS/MAP WGS VCFs are access-controlled through the AD Knowledge Portal/Synapse:

- WGS data: `syn11724057`, <https://www.synapse.org/Synapse:syn11724057>
- underlying DLPFC single-nucleus RNA-seq: `syn31512863`,
  <https://www.synapse.org/Synapse:syn31512863>
- access procedures: <https://adknowledgeportal.synapse.org/Data%20Access>

Qualified investigators must obtain their own approval and comply with data-use terms. Do not copy
individual genotypes or protected scored results into a public worktree.

## Fujita et al. benchmark

Cell-type-level eQTL summary statistics are available at Synapse `syn52335732`:
<https://doi.org/10.7303/syn52335732>.

The benchmark uses Fujita et al.'s published `significant_by_2step_FDR` indicator. It is a useful
external benchmark because it was defined from measured ROS/MAP molecular data independently of
HaploPerturb and AlphaGenome. It should be treated as a published discovery indicator, not as truth
in an absolute or causal sense. The matched-count analyses select the same number of top-scoring
AlphaGenome locus-gene pairs as Fujita positives in the corresponding evaluation family.

## AlphaGenome

AlphaGenome was queried through client version 0.4.0 for human `RNA_SEQ` predictions from
1,048,576-bp GRCh38 inputs. An API credential is required and must be provided through the
`ALPHAGENOME_API_KEY` environment variable. Never store credentials in this repository. The hosted
service did not expose a permanent server-model version identifier, which limits exact future
re-querying.
