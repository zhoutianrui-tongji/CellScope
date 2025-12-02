from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else "CellScope: Spatial transcriptomics pipeline."

setup(
	name="CellScope",
	version="1.0.0",
	description="A modular, resumable spatial transcriptomics pipeline with graph embeddings and annotation",
	long_description=readme,
	long_description_content_type="text/markdown",
	author="CellScope Contributors",
	url="https://github.com/your-org/CellScope",
	license="MIT",
	packages=find_packages(),
	python_requires=">=3.9",
	install_requires=[
		"anndata>=0.10",
		"pandas>=1.5",
		"rich>=13",
		"click>=8",
		"pyarrow>=14",
		"shapely>=2.0",
		"scikit-learn>=1.2",
		"scipy>=1.10",
		"joblib>=1.3",
		"hdbscan>=0.8.33",
		"matplotlib>=3.7",
		"tqdm>=4.65",
		"scanpy>=1.9",
		"statsmodels>=0.14",
		"seaborn>=0.12",
		"umap-learn>=0.5",
		"lightning>=2.1",
		# Optional/accelerated modules
		"torch>=2.1",
		"torch-geometric>=2.5",
		"scvi-tools>=0.20",
	],
	entry_points={
		"console_scripts": [
			"cellscope=cellscope.cli:main",
		]
	},
	classifiers=[
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Bio-Informatics",
	],
)
