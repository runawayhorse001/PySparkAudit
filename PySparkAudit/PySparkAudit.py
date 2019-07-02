import os
import shutil
import time
import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pyspark.sql.functions as F
from pyspark.mllib.stat import Statistics


def mkdir(path):
    """
    Make a new directory. if it's exist, keep the old files.

    :param path: the directory path
    """
    try:
        os.mkdir(path)
    except OSError:
        pass


def mkdir_clean(path):
    """
    Make a new directory. if it's exist, remove the old files.

    :param path: the directory path
    """
    try:
        os.mkdir(path)
    except OSError:
        try:
            if len(os.listdir(path)) != 0:
                shutil.rmtree(path)
                os.mkdir(path)
        except Exception as e:
            print(e)


def df_merge(dfs, key, how='left'):
    """
    Merge multiple pandas data frames with same key.

    :param dfs: name list of the data frames
    :param key: key for join
    :param how: method for join, the default value is left
    :return: merged data frame
    """
    return reduce(lambda left, right: pd.merge(left, right, on=[key], how=how), dfs)


def data_types(df_in, tracking=False):
    """
    Generate the data types of the rdd data frame.

    :param df_in: the input rdd data frame
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: data types pandas data frame
    """
    if tracking:
        print('================================================================')
        print("Collecting data types.... Please be patient!")
        start = time.time()

    d_types = pd.DataFrame(df_in.dtypes, columns=['feature', 'dtypes'])

    if tracking:
        end = time.time()
        print('Generate counts took = ' + str(end - start) + ' s')

    return d_types


def dtypes_class(df_in):
    """
    Generate the data type categories: numerical, categorical, date and unsupported category.

    :param df_in: the input rdd data frame
    :return: data type categories
    """
    # __all__ = [
    # "DataType", "NullType", "StringType", "BinaryType", "BooleanType", "DateType",
    # "TimestampType", "DecimalType", "DoubleType", "FloatType", "ByteType", "IntegerType",
    # "LongType", "ShortType", "ArrayType", "MapType", "StructField", "StructType"]

    # numerical data types in rdd DataFrame dtypes
    num_types = ['DecimalType', 'DoubleType', 'FloatType', 'ByteType', 'IntegerType', 'LongType', 'ShortType']
    # qualitative data types in rdd DataFrame dtypes
    cat_types = ['NullType', 'StringType', 'BinaryType', 'BooleanType']
    # date data types in rdd DataFrame dtypes
    date_types = ['DateType', 'TimestampType']
    # unsupported data types in rdd DataFrame dtypes
    unsupported_types = ['ArrayType', 'MapType', 'StructField', 'StructType']

    all_fields = [(f.name, str(f.dataType)) for f in df_in.schema.fields]

    all_df = pd.DataFrame(all_fields, columns=['feature', 'DataType'])

    # initialize the memory for the corresponding fields
    num_fields = []
    cat_fields = []
    date_fields = []
    unsupported_fields = []

    [num_fields.append(item[0]) if item[1] in num_types else
     cat_fields.append(item[0]) if item[1] in cat_types else
     date_fields.append(item[0]) if item[1] in date_types else
     unsupported_fields.append(item[0]) for item in all_fields]

    return all_df, num_fields, cat_fields, date_fields, unsupported_fields


def counts(df_in, tracking=False):
    """
    Generate the row counts and not null rows and distinct counts for each feature.

    :param df_in: the input rdd data frame
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the counts in pandas data frame
    """
    if tracking:
        print('================================================================')
        print("Collecting features' counts.... Please be patient!")
        start = time.time()

    row_count = df_in.count()

    f_counts = pd.DataFrame([(c, row_count, df_in.na.drop(subset=[c]).select(c).count(),
                              df_in.na.drop(subset=[c]).select(c).distinct().count()) for c in df_in.columns],
                            columns=['feature', 'row_count', 'notnull_count', 'distinct_count'])

    if tracking:
        end = time.time()
        print('Generate counts took = ' + str(end - start) + ' s')

    return f_counts


def describe(df_in, columns=None, tracking=False):
    """
    Generate the simple data frame description using `.describe()` function in pyspark.

    :param df_in: the input rdd data frame
    :param columns: the specific feature columns, the default value is None
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the description in pandas data frame
    """
    if tracking:
        print('================================================================')
        print("Collecting data frame description.... Please be patient!")
        start = time.time()

    if columns:
        df_in = df_in.select(columns)
    desc = df_in.describe().toPandas().set_index('summary').transpose().rename_axis('feature')

    if tracking:
        end = time.time()
        print('Generate data frame description took = ' + str(end - start) + ' s')

    return desc


def percentiles(df_in, deciles=False, tracking=False):
    """
    Generate the percentiles for rdd data frame.

    :param df_in: the input rdd data frame
    :param deciles: the flag for generate the deciles
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: percentiles in pandas data frame
    """
    if tracking:
        print('================================================================')
        print('Calculating percentiles.... Please be patient!')
        start = time.time()

    cols = df_in.columns
    if deciles:
        p_vector = list(np.array(range(0, 110, 10)) / 100)
        names = [str(int(p * 100)) + '%' for p in p_vector]
    else:
        p_vector = [0.25, 0.50, 0.75]
        names = ['Q1', 'Med', 'Q3']

    p = [df_in.approxQuantile(c, p_vector, 0.00) for c in cols]
    p_df = pd.DataFrame(p, columns=names, index=cols).rename_axis('feature').reset_index()

    if tracking:
        end = time.time()
        print('Generate percentiles took = ' + str(end - start) + ' s')

    return p_df


def feature_len(df_in, tracking=False):
    """
    Generate feature length statistical results for each feature in the rdd data frame.

    :param df_in: the input rdd data frame
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the feature length statistical results in pandas data frame
    """
    if tracking:
        print('================================================================')
        print("Calculating features' length.... Please be patient!")
        start = time.time()

    cols = df_in.columns
    features_len = df_in.select(*(F.length(F.col(c)).alias(c) for c in cols))
    summary_len = features_len.agg(*(F.min(F.col(c)).alias(c) for c in cols)) \
        .union(features_len.agg(*(F.avg(F.col(c)).alias(c) for c in cols))) \
        .union(features_len.agg(*(F.max(F.col(c)).alias(c) for c in cols))) \
        .toPandas().transpose().reset_index()
    summary_len.columns = ['feature', 'min_length', 'avg_length', 'max_length']

    if tracking:
        end = time.time()
        print("Generate features' length took = " + str(end - start) + ' s')

    return summary_len


def freq_items(df_in, top_n=5, tracking=False):
    """
    Generate the top_n frequent items in for each feature in the rdd data frame.

    :param df_in: the input rdd data frame
    :param top_n: the number of the most frequent item
    :param tracking: the flag for displaying CPU time, the default value is False
    :return:
    """
    if tracking:
        print('================================================================')
        print('Calculating top {} frequent items.... Please be patient!'.format(top_n))
        start = time.time()

    freq = [[[item[0], item[1]] for item in df_in.groupBy(col).count().sort(F.desc('count')).take(top_n)]
            for col in df_in.columns]

    if tracking:
        end = time.time()
        print('Generate rates took: ' + str(end - start) + ' s')

    return pd.DataFrame({'feature': df_in.columns, 'freq_items[value, freq]': freq})


def rates(df_in, columns=None, numeric=True, tracking=False):
    """
    Generate the null, empty, negative, zero and  positive value rates and feature variance for
    each feature in the rdd data frame.

    :param df_in: the input rdd data frame
    :param columns: the specific feature columns, the default value is None
    :param numeric: the flag for numerical rdd data frame, the default value is True
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the null, empty, negative, zero and  positive value rates and feature variance
             in pandas data frame
    """
    if tracking:
        print('================================================================')
        print('Calculating rates.... Please be patient!')

    if columns is None:
        _, num_fields, cat_fields, _, _ = dtypes_class(df_in)
        columns = num_fields + cat_fields

    start = time.time()
    rate_null = []
    rate_empty = []
    rate_pos = []
    rate_neg = []
    rate_zero = []
    rate_variance = []

    data = df_in.select(columns)
    total = data.count()

    if numeric:
        [(rate_null.append(data.filter(F.col(c).isNull()).count() / total),
          rate_empty.append(data.filter(F.trim(F.col(c)) == '').count() / total),
          rate_pos.append(data.filter(F.col(c) > 0).count() / total),
          rate_neg.append(data.filter(F.col(c) < 0).count() / total),
          rate_zero.append(data.filter(F.col(c) == 0).count() / total),
          rate_variance.append(data.na.drop(subset=[c]).select(c).distinct().count() /
                               data.na.drop(subset=[c]).select(c).count())) for c in data.columns]

        d = {'feature': columns, 'feature_variance': rate_variance, 'rate_null': rate_null,
             'rate_empty': rate_empty, 'rate_neg': rate_neg, 'rate_zero': rate_zero,
             'rate_pos': rate_pos}
    else:
        [(rate_null.append(data.filter(F.col(c).isNull()).count() / total),
          rate_empty.append(data.filter(F.trim(F.col(c)) == '').count() / total),
          rate_variance.append(data.na.drop(subset=[c]).select(c).distinct().count() /
                               data.na.drop(subset=[c]).select(c).count())) for c in data.columns]

        d = {'feature': columns, 'feature_variance': rate_variance, 'rate_null': rate_null,
             'rate_empty': rate_empty}

    if tracking:
        end = time.time()
        print('Generate rates took: ' + str(end - start) + ' s')

    return pd.DataFrame(d)


def corr_matrix(df_in, method="pearson", output_dir=None, rotation=True, display=False, tracking=False):
    """
    Generate the correlation matrix and heat map plot for rdd data frame.

    :param df_in: the input rdd data frame
    :param method: the method which applied to calculate the correlation matrix: pearson or spearman.
                the default value is pearson
    :param output_dir: the out put directory, the default value is the current working directory
    :param rotation: the flag for rotating the xticks in the plot, the default value is True
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the correlation matrix in pandas data frame
    """
    _, num_fields, _, _, _ = dtypes_class(df_in)

    if len(num_fields) > 1:
        df_in = df_in.select(num_fields)
        df_in = df_in.na.drop()

        if output_dir is None:
            out_path = os.getcwd() + '/Audited'
        else:
            out_path = output_dir + '/Audited'
        mkdir(out_path)

        print('================================================================')
        print('The correlation matrix plot Corr.png was located at:')
        print(out_path)
        if tracking:
            print('Calculating correlation matrix... Please be patient!')
            start = time.time()

        # convert the rdd data data frame to dense matrix
        col_names = df_in.columns
        features = df_in.rdd.map(lambda row: row[0:])

        # calculate the correlation matrix
        corr_mat = Statistics.corr(features, method=method)
        corr = pd.DataFrame(corr_mat)
        corr.index, corr.columns = col_names, col_names

        # corr.to_csv('{}/corr_mat.csv'.format(out_path))

        fig = plt.figure(figsize=(20, 15))  # Push new figure on stack
        sns_plot = sns.heatmap(corr, cmap="YlGnBu",
                               xticklabels=corr.columns.values,
                               yticklabels=corr.columns.values)
        if rotation:
            plt.xticks(rotation=90, fontsize=20)
            sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=20)
        plt.savefig("{}/01-correlation_mat.png".format(out_path))
        if display:
            plt.show()
        plt.clf()
        plt.close(fig)
        if tracking:
            end = time.time()
            print('Generate correlation matrix took = ' + str(end - start) + ' s')

        return corr


def hist_plot(df_in, bins=50, output_dir=None, sample_size=None, display=False, tracking=False):
    """
    Histogram plot for the numerical features in the rdd data frame. **This part is super time and
    memory consuming.** If the data size is larger than 10,000, the histograms will be saved in .pdf
    format. Otherwise, the histograms will be saved in .png format in hist folder.

    If your time and memory are limited, you can use sample_size to generate the subset of the data
    frame to generate the histograms.

    :param df_in: the input rdd data frame
    :param bins: the number of bins for generate the bar plots
    :param output_dir: the out put directory, the default value is the current working directory
    :param sample_size: the size for generate the subset from the rdd data frame, the
           default value none
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    """
    _, num_fields, _, _, _ = dtypes_class(df_in)

    if num_fields:
        df_in = df_in.select(num_fields)

        if output_dir is None:
            out_path = os.getcwd() + '/Audited'
        else:
            out_path = output_dir + '/Audited'
        mkdir(out_path)

        if tracking:
            start = time.time()

        if (df_in.count() <= 10000) or (sample_size is not None and sample_size <= 10000):
            print('================================================================')
            print('The Histograms plot Histograms.pdf was located at:')
            print(out_path)
            pdf = PdfPages(out_path + '/02-Histograms.pdf')
            for col in df_in.columns:
                if sample_size is None:
                    data = df_in.select(col).na.drop().toPandas()
                else:
                    data = df_in.select(col).na.drop().toPandas().sample(n=sample_size, random_state=1)

                if tracking:
                    print('Plotting histograms of {}.... Please be patient!'.format(col))

                # Turn interactive plotting off
                plt.ioff()
                fig = plt.figure(figsize=(20, 15))
                sns.distplot(data, bins=bins, kde=False, rug=True)
                plt.title('Histograms of {}'.format(col), fontsize=20)
                plt.xlabel('{}'.format(col), fontsize=20)
                plt.ylabel('number of counts', fontsize=20)
                pdf.savefig(fig)
                if display:
                    plt.show()
                plt.close(fig)
            print('Histograms plots are done!')
            pdf.close()
        else:
            mkdir_clean(out_path + '/02-hist')
            print('================================================================')
            print('The Histograms plots *.png were located at:')
            print(out_path + '/02-hist')
            for col in df_in.columns:
                if sample_size is None:
                    data = df_in.select(col).na.drop().toPandas()
                else:
                    data = df_in.select(col).na.drop().toPandas().sample(n=sample_size, random_state=1)

                if tracking:
                    print('Plotting histograms of {}.... Please be patient!'.format(col))

                fig = plt.figure(figsize=(20, 15))
                sns.distplot(data, bins=bins, kde=False, rug=True)
                plt.title('Histograms of {}'.format(col), fontsize=20)
                plt.xlabel('{}'.format(col), fontsize=20)
                plt.ylabel('number of counts', fontsize=20)
                plt.savefig(out_path + '/02-hist/' + "{}.png".format(col))
                if display:
                    plt.show()
                plt.clf()
                plt.close(fig)
            if tracking:
                print('Histograms plots are DONE!!!')

        if tracking:
            end = time.time()
            print('Generate histograms plots took = ' + str(end - start) + ' s')
    else:
        print('Caution: no numerical features in the dataset!!!')


def bar_plot(df_in, top_n=20, rotation=True, output_dir=None, display=False, tracking=False):
    """
    Bar plot for the categorical features in the rdd data frame.

    :param df_in: the input rdd data frame
    :param top_n: the number of the most frequent feature to show in the bar plot
    :param rotation: the flag for rotating the xticks in the plot, the default value is True
    :param output_dir: the out put directory, the default value is the current working directory
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    """
    _, _, cat_fields, date_fields, _ = dtypes_class(df_in)

    cat_fields = cat_fields + date_fields
    if cat_fields:
        df_in = df_in.select(cat_fields)

        if output_dir is None:
            out_path = os.getcwd() + '/Audited'
        else:
            out_path = output_dir + '/Audited'
        mkdir(out_path)

        print('================================================================')
        print('The Bar plot Bar_plots.pdf was located at:')
        print(out_path)
        if tracking:
            start = time.time()

        pdf = PdfPages(out_path + '/03-Bar_plots.pdf')
        for col in df_in.columns:
            p_data = df_in.select(col).na.drop().groupBy(col).count().sort(F.desc('count')).limit(top_n).toPandas()

            if tracking:
                print('Plotting barplot of {}.... Please be patient!'.format(col))
            plt.ioff()
            fig = plt.figure(figsize=(20, 15))
            sns.barplot(x=col, y="count", data=p_data)
            plt.title('Barplot of {}'.format(col), fontsize=20)
            plt.xlabel('{}'.format(col), fontsize=20)
            plt.ylabel('number of counts', fontsize=20)
            if rotation:
                plt.xticks(rotation=90)
            pdf.savefig(fig)
            if display:
                plt.show()
            plt.close(fig)
        if tracking:
            print('Bar plots are DONE!!!')
        pdf.close()

        if tracking:
            end = time.time()
            print('Generate bar plots took = ' + str(end - start) + ' s')
    else:
        print('Caution: no categorical features in the dataset!!!')


def trend_plot(df_in, types='day', d_time=None, rotation=True, output_dir=None, display=False, tracking=False):
    """
    Trend plot for the aggregated time series data if the rdd data frame has date features and numerical features.

    :param df_in: the input rdd data frame
    :param types: the types for time feature aggregation: day, month, year, the default value is day
    :param d_time: the specific feature name of the date feature, the default value
                   is the first date feature in the rdd data frame
    :param rotation: the flag for rotating the xticks in the plot, the default value is True
    :param output_dir: the out put directory, the default value is the current working directory
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    """
    _, num_fields, _, date_fields, _ = dtypes_class(df_in)

    if date_fields and num_fields:

        df_in = df_in.select(date_fields+num_fields)

        if d_time is None:
            d_time = date_fields[0]

        if output_dir is None:
            out_path = os.getcwd() + '/Audited'
        else:
            out_path = output_dir + '/Audited'
        mkdir(out_path)

        print('================================================================')
        print('The Trend plot Trend_plots.pdf was located at:')
        print(out_path)

        if tracking:
            start = time.time()

        pdf = PdfPages(out_path + '/04-Trend_plots.pdf')
        if types == 'day':
            ts_format = 'yyyy-MM-dd'
        elif types == 'month':
            ts_format = 'yyyy-MM'
        elif types == 'year':
            ts_format = 'yyyy'

        for col in num_fields:

            p_data = df_in.select(F.date_format(d_time, ts_format).alias(types), col) \
                .groupBy(types).agg(F.mean(col).alias("mean"), F.sum(col).alias("sum")).toPandas()

            if tracking:
                print('Plotting trend plot of {}.... Please be patient!'.format(col))

            plt.ioff()
            sns.set(style="ticks", rc={"lines.linewidth": 2})
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            sns.lineplot(x=types, y="mean", data=p_data, ax=axes[0])
            axes[0].set_title('Mean trend of {} in {}'.format(col, types))
            sns.lineplot(x=types, y="sum", data=p_data, ax=axes[1])
            axes[1].set_title('Sum trend of {} in {}'.format(col, types))

            if rotation:
                for ax in fig.axes:
                    plt.sca(ax)
                    plt.xticks(rotation=90, fontsize=8)

            pdf.savefig(fig)

            if display:
                plt.show()
            plt.close(fig)
        if tracking:
            print('Trend plots are DONE!!!')
        pdf.close()

        if tracking:
            end = time.time()
            print('Generate trend plots took = ' + str(end - start) + ' s')
    else:
        print('Caution: no date features in the dataset!!!')


def dataset_summary(df_in, tracking=False):
    """
    The data set basics summary.

    :param df_in: the input rdd data frame
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: data set summary in pandas data frame
    """
    if tracking:
        start = time.time()

    sample_size = df_in.count()

    col_names = ['summary', 'value']
    all_fields, num_fields, cat_fields, date_fields, unsupported_fields = dtypes_class(df_in)
    d_types = all_fields.groupby('DataType').count().reset_index()
    d_types.columns = col_names

    cols = df_in.columns
    mask = df_in.withColumn('zero_count', sum(F.when(F.col(c) == '0', 1).otherwise(0) for c in cols)) \
                .withColumn('null_count', sum(F.col(c).isNull().cast('int') for c in cols)) \
                .withColumn('empty_count', sum(F.when(F.trim(F.col(c)) == '', 1).otherwise(0) for c in cols)) \
                .select(['null_count', 'empty_count', 'zero_count'])

    row_w_null = mask.filter(F.col('null_count') > 0).count()
    row_w_empty = mask.filter(F.col('empty_count') > 0).count()
    row_w_zero = mask.filter(F.col('zero_count') > 0).count()

    r_avg = mask.agg(*[F.avg(c).alias('row_avg_' + c) for c in mask.columns]).toPandas().transpose().reset_index()
    r_avg.columns = col_names

    single_unique_feature = sum([df_in.na.drop(subset=[c]).select(c).distinct().count() == 1 for c in cols])

    size_names = ['sample_size', 'feature_size', 'single_unique_feature', 'row_w_null', 'row_w_empty', 'row_w_zero']
    size_values = [sample_size, len(cols), single_unique_feature, row_w_null, row_w_empty, row_w_zero]
    size_summary = pd.DataFrame({'summary': size_names, 'value': size_values})

    avg_summary = r_avg

    field_names = ['numerical_fields', 'categorical_fields', 'date_fields', 'unsupported_fields']
    field_values = [len(num_fields), len(cat_fields), len(date_fields), len(unsupported_fields)]
    field_summary = pd.DataFrame({'summary': field_names, 'value': field_values})

    types_summary = pd.DataFrame(df_in.dtypes, columns=['value','dtypes'])\
                      .groupby('dtypes').count().rename_axis('summary').reset_index()

    summary = pd.concat([size_summary, avg_summary, field_summary, types_summary], axis=0)

    if tracking:
        end = time.time()
        print('Generate data set summary took = ' + str(end - start) + ' s')

    return summary


def numeric_summary(df_in, columns=None, deciles=False, top_n=5, tracking=False):
    """
    The auditing function for numerical rdd data frame.

    :param df_in: the input rdd data frame
    :param columns: the specific feature columns, the default value is None
    :param deciles: the flag for generate the deciles
    :param top_n: the number of the most frequent item
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the audited results for the numerical features in pandas data frame
    """
    _, num_fields, _, _, _ = dtypes_class(df_in)

    if num_fields:
        if tracking:
            start = time.time()

        num = df_in.select(num_fields)
        if columns:
            num = num.select(columns)

        d_types = data_types(num, tracking=tracking)
        f_counts = counts(num, tracking=tracking)
        des = describe(num, columns=columns, tracking=tracking)
        p_df = percentiles(num, deciles=deciles, tracking=tracking)
        f_len = feature_len(num, tracking=tracking)
        freq = freq_items(num, top_n=top_n, tracking=tracking)
        rate = rates(num, columns=columns, numeric=True, tracking=tracking)

        data_frames = [d_types, f_counts, des, p_df, f_len, freq, rate]
        num_summary = df_merge(data_frames, 'feature').drop(['count'], axis=1)

        if tracking:
            end = time.time()
            print('Auditing numerical data took = ' + str(end - start) + ' s')

        return num_summary
    else:
        print('Caution: no numerical features in the dataset!!!')


def category_summary(df_in, columns=None, top_n=5, tracking=False):
    """
    The auditing function for categorical rdd data frame.

    :param df_in: the input rdd data frame
    :param columns: the specific feature columns, the default value is None
    :param top_n: the number of the most frequent item
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the audited results for the categorical features in pandas data frame
    """
    _, _, cat_fields, _, _ = dtypes_class(df_in)

    if cat_fields:
        if tracking:
            start = time.time()

        cat = df_in.select(cat_fields)
        if columns:
            cat = cat.select(columns)

        d_types = data_types(cat, tracking=tracking)
        f_counts = counts(cat,tracking=tracking)
        f_len = feature_len(cat, tracking=tracking)
        freq = freq_items(cat, top_n=top_n, tracking=tracking)
        rate = rates(cat, columns=columns, numeric=False, tracking=tracking)

        data_frames = [d_types, f_counts, f_len, freq, rate]
        cat_summary = df_merge(data_frames, 'feature')

        if tracking:
            end = time.time()
            print('Auditing categorical data took = ' + str(end - start) + ' s')

        return cat_summary
    else:
        print('Caution: no numerical features in the dataset!!!')


def fig_plots(df_in, output_dir=None, bins=50, top_n=20, types='day', d_time=None,
              rotation=True, sample_size=None, display=False, tracking=False):
    """
    The wrapper for the plot functions.

    :param df_in: the input rdd data frame
    :param output_dir: the out put directory, the default value is the current working directory
    :param bins: the number of bins for generate the bar plots
    :param top_n: the number of the most frequent feature to show in the bar plot
    :param types: the types for time feature aggregation: day, month, year, the default value is day
    :param d_time: the specific feature name of the date feature, the default value
                   is the first date feature in the rdd data frame
    :param rotation: the flag for rotating the xticks in the plot, the default value is True
    :param sample_size: the size for generate the subset from the rdd data frame, the
           default value none
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    """
    if tracking:
        start = time.time()

    hist_plot(df_in, bins=bins, output_dir=output_dir, sample_size=sample_size, display=display, tracking=tracking)
    bar_plot(df_in, top_n=top_n, rotation=rotation, output_dir=output_dir, display=display, tracking=tracking)
    trend_plot(df_in, types=types, d_time=d_time, rotation=rotation, output_dir=output_dir,
               display=display, tracking=tracking)

    if tracking:
        end = time.time()
        print('Generate all the figures took = ' + str(end - start) + ' s')


def auditing(df_in, writer=None, columns=None, deciles=False, top_freq_item=5, bins=50, top_cat_item=20,
             method="pearson", output_dir=None, types='day', d_time=None, rotation=True, sample_size=None,
             display=False, tracking=False):
    """
    The wrapper of auditing functions.

    :param df_in: the input rdd data frame
    :param writer: the writer for excel output
    :param columns: the specific feature columns, the default value is None
    :param deciles: the flag for generate the deciles
    :param top_freq_item: the number of the most frequent item
    :param bins: the number of bins for generate the bar plots
    :param top_cat_item: the number of the most frequent feature to show in the bar plot
    :param method: the method which applied to calculate the correlation matrix: pearson or spearman.
                the default value is pearson
    :param output_dir: the out put directory, the default value is the current working directory
    :param types: the types for time feature aggregation: day, month, year, the default value is day
    :param d_time: the specific feature name of the date feature, the default value
                   is the first date feature in the rdd data frame
    :param rotation: the flag for rotating the xticks in the plot, the default value is True
    :param sample_size: the size for generate the subset from the rdd data frame, the
           default value none
    :param display: the flag for displaying the figures, the default value is False
    :param tracking: the flag for displaying CPU time, the default value is False
    :return: the all audited results in pandas data frame
    """
    if output_dir is None:
        out_path = os.getcwd() + '/Audited'
    else:
        out_path = output_dir + '/Audited'
    mkdir_clean(out_path)

    if writer is None:
        writer = pd.ExcelWriter(out_path + '/00-audited_results.xlsx', engine='xlsxwriter')
    else:
        writer = writer

    print('================================================================')
    print('The audited results summary audited_results.xlsx was located at:')
    print(out_path)

    start = time.time()

    all_fields, num_fields, cat_fields, _, _ = dtypes_class(df_in)

    if all_fields.shape[0] > 1:
        set_summary = dataset_summary(df_in, tracking=tracking)
        set_summary.to_excel(writer, sheet_name='Dataset_summary', index=False)
    if num_fields:
        num_summary = numeric_summary(df_in, columns=columns, deciles=deciles, top_n=top_freq_item, tracking=tracking)
        num_summary.to_excel(writer, sheet_name='Numeric_summary', index=False)
    else:
        num_summary = []
    if cat_fields:
        cat_summary = category_summary(df_in, columns=columns, top_n=top_freq_item, tracking=tracking)
        cat_summary.to_excel(writer, sheet_name='Category_summary', index=False)
    else:
        cat_summary = []
    if len(num_fields) > 1:
        corr = corr_matrix(df_in, method=method, output_dir=output_dir, rotation=rotation,
                           display=display, tracking=tracking)
        corr.to_excel(writer, sheet_name='Correlation_matrix', index=False)
    else:
        corr = []

    writer.save()

    fig_plots(df_in, output_dir=output_dir, bins=bins, top_n=top_cat_item, types=types, d_time=d_time,
              rotation=rotation, sample_size=sample_size, display=display, tracking=tracking)

    end = time.time()
    print('Generate all audited results took = ' + str(end - start) + ' s')
        
    print('================================================================')
    print('The auditing processes are DONE!!!')

    if display:
        return num_summary, cat_summary, corr




