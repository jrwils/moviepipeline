
import os
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    IntegerType,
    StringType,
    ArrayType
)
from pyspark.sql.functions import lit


IMDB_SOURCE_DIR = "s3a://moviepipeline/imdb/"
ML_SOURCE_DIR = "s3a://moviepipeline/grouplens/"
OUTPUT_DIR = "s3a://moviepipeline/output/"
CLUSTER_PARTITIONS = 3


@udf(IntegerType())
def parse_year(orig_title):
    """
    Parses the year from the ending parentheses of a title.
    For example, 'Toy Story (1995)' will return 1995.
    """
    year = orig_title[orig_title.rfind("(") + 1: orig_title.rfind(")")]
    if year.isnumeric():
        return int(year)
    return None


@udf(StringType())
def parse_title(orig_title):
    """
    Parses the title from the beginning of a title with
    the year in parentheses.
    For example, 'Toy Story (1995)' will return 'Toy Story'.
    """
    return orig_title[: orig_title.rfind("(")].strip()


@udf(ArrayType(StringType()))
def parse_genres(orig_genres, delimiter):
    """
    Parses a string into a list/array based on the
    given delimiter.
    For example,
        orig_genres='Documentary, Short',
        delimiter=','
    will return ['Documentary', 'Short']
    """
    return orig_genres.split(delimiter)


@udf(StringType())
def parse_numeric(id_number):
    """
    Returns a string of only numeric characters for a string.
    For example, id_number='tt0089808' will return '0089808'
    """
    return ''.join([i for i in id_number if i.isnumeric()])


class MoviePipeline(object):
    def __init__(self):
        # Create the Spark session to be used throughout
        # the pipeline process
        self.spark = SparkSession \
            .builder \
            .config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.0.0"
            ) \
            .getOrCreate()

        # Initialize and inventory the DataFrames that will
        # be used in the pipeline

        # IMDB source data
        self.imdb_title_basics_df = None
        self.imdb_name_basics_df = None
        self.imdb_title_prin_df = None
        self.imdb_title_crew_df = None
        self.imdb_title_ratings_df = None

        # MovieLens source data
        self.ml_movies_df = None
        self.ml_links_df = None
        self.ml_tags_df = None
        self.ml_ratings_df = None
        self.ml_genome_scores_df = None
        self.ml_genome_tags_df = None

        # Data model
        self.movie = None
        self.moviegenre = None
        self.name = None
        self.castcrew = None
        self.usertag = None
        self.genometag = None

    def read_imdb_source_files(self):
        """
        Read the IMDB source .tsv files
        into DataFrame instances
        """
        imdb_read_options = {
            'sep': '\t',
            'header': True
        }
        self.imdb_title_basics_df = self.spark.read.options(
            **imdb_read_options).csv(
            os.path.join(IMDB_SOURCE_DIR, "title.basics.tsv.gz")
        )

        self.imdb_name_basics_df = self.spark.read.options(
            **imdb_read_options).csv(
            os.path.join(IMDB_SOURCE_DIR, "name.basics.tsv.gz")
        )

        self.imdb_title_prin_df = self.spark.read.options(
            **imdb_read_options).csv(
            os.path.join(IMDB_SOURCE_DIR, "title.principals.tsv.gz")
        )

        self.imdb_title_crew_df = self.spark.read.options(
            **imdb_read_options).csv(
            os.path.join(IMDB_SOURCE_DIR, "title.crew.tsv.gz")
        )

        self.imdb_title_ratings_df = self.spark.read.options(
            **imdb_read_options).csv(
            os.path.join(IMDB_SOURCE_DIR, "title.ratings.tsv.gz")
        )

    def read_movielens_source_files(self):
        """
        Read the MovieLens source .csv files
        into DataFrame instances
        """
        self.ml_movies_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "movies.csv"))

        self.ml_links_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "links.csv"))

        self.ml_tags_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "tags.csv"))

        self.ml_ratings_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "ratings.csv"))

        self.ml_genome_scores_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "genome-scores.csv"))

        self.ml_genome_tags_df = self.spark.read.options(
            header=True).csv(os.path.join(ML_SOURCE_DIR, "genome-tags.csv"))

    def clean_imdb_title_basics(self):
        """
        The IMDB title.basics.tsv file contains more than movies.
        Since we are only interested in movies, we will filter
        out everything else.
        """
        self.imdb_title_basics_df = self.imdb_title_basics_df.where(
            "titleType = 'movie'"
        )

    def clean_ml_movies_title(self):
        """
        The MovieLens movies file contains titles with the year in parentheses.
        For example, 'Toy Story (1995)'. This method parses the title and
        year values and puts them into separate fields of the DataFrame.
        """
        self.ml_movies_df = self.ml_movies_df.withColumnRenamed(
            "title", "origTitle"
        )
        self.ml_movies_df = self.ml_movies_df.withColumn(
            "title", parse_title("origTitle")
        )
        self.ml_movies_df = self.ml_movies_df.withColumn(
            "year", parse_year("origTitle")
        )
        self.ml_movies_df = self.ml_movies_df.drop("origTitle")

    def clean_ml_movies_genres(self):
        """
        The MovieLens movies file contains genres that are
        pipe-separated.
        For example, 'Adventure|Animation|Children|Comedy|Fantasy'.
        This method splits them into an array structure.
        """
        self.ml_movies_df = self.ml_movies_df.withColumnRenamed(
            "genres", "origGenres"
        )
        self.ml_movies_df = self.ml_movies_df.withColumn(
            "genres", parse_genres("origGenres", lit("|"))
        )
        self.ml_movies_df = self.ml_movies_df.drop("origGenres")

    def clean_imdb_title_ids(self):
        """
        This method removes non-numeric characters from the 'tconst'
        ID numbers that represent IMDB movie titles.
        """
        self.imdb_title_basics_df = self.imdb_title_basics_df.withColumn(
            "tconst",
            parse_numeric("tconst")
        )
        self.imdb_title_prin_df = self.imdb_title_prin_df.withColumn(
            "tconst",
            parse_numeric("tconst")
        )
        self.imdb_title_ratings_df = self.imdb_title_ratings_df.withColumn(
            "tconst",
            parse_numeric("tconst")
        )

    def clean_imdb_name_ids(self):
        """
        This method removes non-numeric characters from the 'nconst'
        ID numbers that represent IMDB names.
        """
        self.imdb_name_basics_df = self.imdb_name_basics_df.withColumn(
            "nconst",
            parse_numeric("nconst")
        )
        self.imdb_title_prin_df = self.imdb_title_prin_df.withColumn(
            "nconst",
            parse_numeric("nconst")
        )

    def clean_imdb_genres(self):
        """
        The IMDB title.basics.tsv file contains genres that are
        comma-separated.
        For example, 'Documentary,Short'.
        This method splits them into an array structure.
        """
        self.imdb_title_basics_df = self.imdb_title_basics_df \
            .withColumnRenamed(
                "genres",
                "origGenres"
            )
        self.imdb_title_basics_df = self.imdb_title_basics_df.withColumn(
            "genres",
            parse_genres("origGenres", lit(","))
        )
        self.imdb_title_basics_df = self.imdb_title_basics_df.drop(
            "origGenres"
        )

    def clean_movielens_ratings(self):
        """
        The MovieLens ratings.csv file is grouped into
        individual ratings by user.
        This method averages and aggregates them by movie title.
        """
        self.ml_ratings_df.createOrReplaceTempView("ml_ratings")
        self.ml_ratings_df = self.spark.sql(
            """
            SELECT
                movieId,
                avg(rating) as average_rating,
                count(movieId) as ratings
            FROM ml_ratings
            GROUP BY movieId
            """
        )

    def create_movie_df(self):
        """
        Create the movie table in the model
        """
        self.movie = self.imdb_title_basics_df.selectExpr(
            "tconst as imdb_id",
            "primaryTitle as title",
            "startYear as release_year",
            "runtimeMinutes as length_minutes",
            "isAdult as is_adult"
        )

    def populate_movie_ratings(self):
        """
        Insert the movie rating scores from both the IMDB and
        MovieLens source files, along with the number of ratings
        from each source.
        """
        self.movie.createOrReplaceTempView("movie")
        self.imdb_title_ratings_df.createOrReplaceTempView("imdb_ratings")
        self.ml_links_df.createOrReplaceTempView("ml_links")
        self.ml_movies_df.createOrReplaceTempView("ml_movies")
        self.ml_ratings_df.createOrReplaceTempView("ml_ratings")

        # join the data from both sources
        self.movie = self.spark.sql(
            """
            SELECT
                m.imdb_id,
                m.title,
                m.release_year,
                m.length_minutes,
                m.is_adult,
                CAST(ir.averageRating  as double) as imdb_avg_rating,
                CAST(numVotes as int) as imdb_num_ratings,
                CAST(mlr.average_rating as double) as ml_avg_rating,
                CAST(mlr.ratings as int) as ml_num_ratings
            FROM movie m
            LEFT OUTER JOIN imdb_ratings ir ON (m.imdb_id = ir.tconst)
            LEFT OUTER JOIN ml_links mll ON (m.imdb_id = mll.imdbId)
            LEFT OUTER JOIN ml_movies mlm ON (mll.movieId = mlm.movieId)
            LEFT OUTER JOIN ml_ratings mlr ON (mlr.movieId = mlm.movieId)
            """
        )

    def calculate_movie_ratings(self):
        """
        Calculate the weighted average between the two sets of ratings
        from each source. This will populate the average_rating
        field of the movie table.
        """
        self.movie.createOrReplaceTempView("movie")
        self.movie = self.spark.sql(
            """
            SELECT
                m.imdb_id,
                m.title,
                m.release_year,
                m.length_minutes,
                m.is_adult,

                CASE
                WHEN
                    imdb_num_ratings is NOT NULL AND
                    imdb_avg_rating is NOT NULL AND
                    ml_num_ratings is NOT NULL AND
                    ml_avg_rating is NOT NULL
                THEN
                ROUND(
                    (imdb_avg_rating *
                        (imdb_num_ratings/(imdb_num_ratings + ml_num_ratings))
                    ) +
                    ((ml_avg_rating * 2) *
                        (ml_num_ratings/(imdb_num_ratings + ml_num_ratings))
                    )
                    , 1
                )
                WHEN
                    imdb_num_ratings IS NOT NULL AND
                    imdb_avg_rating IS NOT NULL AND
                    ml_num_ratings IS NULL AND
                    ml_avg_rating IS NULL
                THEN
                    ROUND(imdb_avg_rating, 1)
                WHEN
                    imdb_num_ratings IS NULL AND
                    imdb_avg_rating IS NULL AND
                    ml_num_ratings IS NOT NULL AND
                    ml_avg_rating IS NOT NULL
                THEN
                    ROUND(ml_avg_rating, 1)
                ELSE null
                END
                as average_rating
            from movie m
            """
        )

    def create_moviegenre_df(self):
        """
        Create the moviegenre table by combining the genres
        from IMDB and MovieLens.
        """
        self.imdb_title_basics_df.createOrReplaceTempView("imdb_title_basics")
        self.moviegenre = self.spark.sql(
            """
            SELECT * from (
                SELECT
                    tconst as imdb_id,
                    EXPLODE(genres) as genre
                FROM imdb_title_basics

                UNION

                SELECT
                    mll.imdbId as imdb_id,
                    EXPLODE(genres) as genre
                FROM ml_movies mlm
                INNER JOIN ml_links mll on (mlm.movieId = mll.movieId)
            )
            WHERE genre != '\\\\N'
            """
        )

    def create_name_df(self):
        """
        Create the name table from the IMDB source data.
        """
        self.imdb_name_basics_df.createOrReplaceTempView("imdb_name_basics")
        self.name = self.spark.sql(
            """
            SELECT
                nconst as name_id,
                primaryName as name,
                birthYear as birth_year,
                deathYear as death_year
            FROM imdb_name_basics
            """
        )

    def create_castcrew_df(self):
        """
        Create the castcrew table from IMDB source data,
        with a relation to the name table.
        """
        self.imdb_title_prin_df.createOrReplaceTempView(
            "imdb_title_principals"
        )
        self.castcrew = self.spark.sql(
            """
            SELECT
                nconst as name_id,
                tconst as imdb_id,
                category
            FROM imdb_title_principals
            """
        )

    def create_usertag_df(self):
        """
        Create the usertag table based on the MovieLens
        user-tagged data.
        """
        self.ml_tags_df.createOrReplaceTempView("ml_tags")
        self.usertag = self.spark.sql(
            """
            SELECT
                mll.imdbId as imdb_id,
                tag,
                timestamp
            FROM ml_tags mlt
            INNER JOIN ml_links mll ON (mll.movieId = mlt.movieId)
            """
        )

    def create_genometag_df(self):
        """
        Create the genometag table based on the MovieLens
        genome tags and scores.
        """
        self.ml_genome_scores_df.createOrReplaceTempView("ml_genome_scores")
        self.ml_genome_tags_df.createOrReplaceTempView("ml_genome_tags")

        self.genometag = self.spark.sql(
            """
            SELECT
                mll.imdbId as imdb_id,
                mlgt.tag,
                CAST(mlgs.relevance as FLOAT) as relevance
            FROM ml_genome_scores mlgs
            INNER JOIN ml_genome_tags mlgt on (mlgs.tagId = mlgt.tagId)
            INNER JOIN ml_links mll ON (mlgs.movieId = mll.movieId)
            """
        )

    def repartition_model(self):
        """
        Repartition the data by imdb_id and name_id
        to avoid shuffling when querying the data.
        """
        num_partitions = CLUSTER_PARTITIONS
        self.movie = self.movie.repartitionByRange(
            num_partitions,
            "imdb_id"
        )
        self.moviegenre = self.moviegenre.repartitionByRange(
            num_partitions,
            "imdb_id"
        )
        self.castcrew = self.castcrew.repartitionByRange(
            num_partitions,
            "imdb_id",
            "name_id"
        )
        self.name = self.name.repartitionByRange(
            num_partitions,
            "name_id"
        )
        self.usertag = self.usertag.repartitionByRange(
            num_partitions,
            "imdb_id"
        )
        self.genometag = self.genometag.repartitionByRange(
            num_partitions,
            "imdb_id"
        )

    def test_movie_null_key(self):
        """
        Test that the movie table does not contain
        any blank or NULL ID values
        """
        movie_null_key_check = self.movie.where(
            "imdb_id = '' or imdb_id is NULL"
        )
        if movie_null_key_check.count() > 0:
            raise ValueError("The movie table contains blank or NULL values.")

    def test_movie_unique_key(self):
        """
        Test that the movie table does not contain any
        duplicate imdb_id values
        """
        self.movie.createOrReplaceTempView("movie")
        movie_unique_key_check = self.spark.sql(
            """
            SELECT
                imdb_id,
                count(imdb_id)
            FROM movie group by imdb_id having count(imdb_id) > 1
            """
        )
        if movie_unique_key_check.count() > 0:
            raise ValueError(
                "The movie table contains duplicate imdb_id values."
            )

    def test_name_null_key(self):
        """
        Test that the name table does not contain any blank or NULL ID values
        """
        name_null_key_check = self.name.where(
            "name_id = '' or name_id is NULL"
        )
        if name_null_key_check.count() > 0:
            raise ValueError("The name table contains blank or NULL values.")

    def test_name_unique_key(self):
        """
        Test that the movie table does not contain any
        duplicate imdb_id values
        """
        self.name.createOrReplaceTempView("name")
        name_unique_key_check = self.spark.sql(
            """
            SELECT
                name_id,
                count(name_id)
            FROM name
            GROUP BY name_id having count(name_id) > 1
            """
        )
        if name_unique_key_check.count() > 0:
            raise ValueError(
                "The name table contains duplicate name_id values."
            )

    def test_movie_count(self):
        """
        Test that the source IMDB movie file contains
        the same amount of records as in the movie table
        """
        imdb_source_count = self.imdb_title_basics_df.count()
        movie_count = self.movie.count()
        if imdb_source_count != movie_count:
            raise ValueError(
                f"""
                Count mismatch - the movie table contains {movie_count} records
                and the IMDB source file contains {imdb_source_count} records."
                """
            )

    def test_name_count(self):
        """
        Test that the source IMDB name file contains the
        same amount of records as in the name table
        """
        imdb_source_count = self.imdb_name_basics_df.count()
        name_count = self.name.count()
        if imdb_source_count != name_count:
            raise ValueError(
                f"""
                Count mismatch - the name table contains {name_count} records
                and the IMDB source file contains {imdb_source_count} records.
                """
            )

    def write_model(self):
        """
        Write the model's output to parquet files on S3.
        """
        self.movie.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "movie.parquet"))
        self.moviegenre.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "moviegenre.parquet"))
        self.name.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "name.parquet"))
        self.castcrew.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "castcrew.parquet"))
        self.usertag.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "usertag.parquet"))
        self.genometag.write.mode("overwrite") \
            .parquet(os.path.join(OUTPUT_DIR, "genometag.parquet"))


def run():
    pipeline = MoviePipeline()

    logging.info("Loading the IMDB data")
    pipeline.read_imdb_source_files()

    logging.info("Loading the MovieLens data")
    pipeline.read_movielens_source_files()

    logging.info("Running cleaning steps")
    pipeline.clean_imdb_title_basics()
    pipeline.clean_ml_movies_title()
    pipeline.clean_ml_movies_genres()
    pipeline.clean_imdb_title_ids()
    pipeline.clean_imdb_name_ids()
    pipeline.clean_imdb_genres()
    pipeline.clean_movielens_ratings()

    logging.info("Creating the Data Model")
    logging.info("Creating the movie DataFrame")
    pipeline.create_movie_df()
    logging.info("Populating movie ratings")
    pipeline.populate_movie_ratings()
    logging.info("Calculating average movie ratings")
    pipeline.calculate_movie_ratings()
    logging.info("Creating the moviegenre DataFrame")
    pipeline.create_moviegenre_df()
    logging.info("Creating the name DataFrame")
    pipeline.create_name_df()
    logging.info("Creating the castcrew DataFrame")
    pipeline.create_castcrew_df()
    logging.info("Creating the usertag DataFrame")
    pipeline.create_usertag_df()
    logging.info("Creating the genometag DataFrame")
    pipeline.create_genometag_df()

    logging.info("Repartitioning the model DataFrames")
    pipeline.repartition_model()

    logging.info("Running Data Quality Checks")
    pipeline.test_movie_null_key()
    pipeline.test_movie_unique_key()
    pipeline.test_name_null_key()
    pipeline.test_name_unique_key()
    pipeline.test_movie_count()
    pipeline.test_name_count()

    logging.info("Writing output files")
    pipeline.write_model()


if __name__ == '__main__':
    run()
