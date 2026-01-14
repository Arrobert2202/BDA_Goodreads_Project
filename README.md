# BDA Project - Goodreads Books Analysis

A Big Data Analytics project for analyzing book data from Goodreads using Apache Spark and OpenSearch.

## 📖 Overview

This project performs exploratory data analysis (EDA), clustering, and data indexing on a large-scale Goodreads books dataset containing over **1 million books**. The analysis pipeline leverages PySpark for distributed processing and OpenSearch for data indexing and search capabilities.

## 📊 Dataset

The dataset contains book metadata with the following key attributes:
- `book_id` - Unique book identifier
- `title` - Book title
- `authors` - List of authors with roles
- `description` - Book description text
- `average_rating` - Average user rating
- `num_pages` - Number of pages
- `ratings_count` - Total number of ratings
- `text_reviews_count` - Number of text reviews
- `popular_shelves` - User-generated tags/genres
- `series` - Series information
- `similar_books` - Related book recommendations

## 📁 Repository Structure

```
BDA_Goodreads_Project/
│
├── README.md                          # Project documentation
│
├── EDA/                               # Exploratory Data Analysis
│   ├── book.json                      # Sample book JSON schema
│   ├── statistics.py                  # Statistical analysis script (PySpark)
│   ├── clusters.py                    # Clustering algorithms (KMeans, LDA)
│   └── results/                       # Analysis output files
│       ├── parquet_stats.txt          # Full dataset statistics results
│       ├── parquet_clustering.txt     # Full dataset clustering results
│       ├── sample_stats.txt           # Sample dataset statistics
│       └── sample_clustering.txt      # Sample dataset clustering
│
└── opensearch_scripts/                # OpenSearch integration
    ├── analyze_hdfs_parquet.py        # HDFS Parquet data inspection utility
    └── import_data_opensearch.py      # Bulk data indexing to OpenSearch
```

## 🔧 Components

### EDA Scripts

#### `statistics.py`
Comprehensive statistical analysis using PySpark:
- Book count summaries and missing data analysis
- Page count statistics (mean, median, quartiles)
- Page distribution categorization
- Series analysis
- Rating distribution analysis
- Author and genre statistics

#### `clusters.py`
Machine learning clustering approaches:
- **Shelf-based Topic Extraction** - Extracts topics from user-generated shelves/tags
- **Keyword-based Clustering** - Clusters books by presence of predefined keywords in descriptions
- Uses TF-IDF vectorization, KMeans, and LDA algorithms

### OpenSearch Scripts

#### `analyze_hdfs_parquet.py`
Utility script for inspecting Parquet data stored in HDFS:
- Inspects NLP processed data
- Inspects Recommender system data
- Inspects EDA processed data

#### `import_data_opensearch.py`
Bulk data indexing script for OpenSearch:
- Indexes book metadata for search functionality
- Uses bulk API for efficient data loading
- Indexes title, description, ratings, and similar books

## 🚀 Technologies

- **Apache Spark (PySpark)** - Distributed data processing
- **HDFS** - Distributed file storage
- **OpenSearch** - Search and analytics engine
- **Python** - Primary programming language

## 📈 Key Statistics (Full Dataset)

| Metric | Value |
|--------|-------|
| Total Books | 1,058,237 |
| Books with Descriptions | 92.85% |
| Books in Series | 36.86% |
| Average Page Count | 278 pages |
| Median Page Count | 259 pages |

## 🛠️ Usage

### Prerequisites
- Apache Spark with PySpark
- HDFS cluster access
- OpenSearch instance (for indexing scripts)

### Running Statistical Analysis
```bash
spark-submit EDA/statistics.py
```

### Running Clustering Analysis
```bash
spark-submit EDA/clusters.py
```

### Indexing Data to OpenSearch
```bash
spark-submit opensearch_scripts/import_data_opensearch.py
```

## 📄 License

This project is for educational purposes as part of a Big Data Analytics course.
