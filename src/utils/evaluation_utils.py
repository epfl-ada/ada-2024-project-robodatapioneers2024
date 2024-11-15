import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def diff_in_diff(df_orig, y_column, x_column, event_date, event_name, offset_months=4):
    # Column for event (0 before the event, 1 after the event)
    df = df_orig[[y_column, x_column, "category"]].copy()

    df.set_index(y_column, inplace=True)
    df.index = pd.to_datetime(df.index)

    # df = df.groupby(df.index.date).sum()
    # is is correct to make a sum here, or would it be better to do a mean?
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    
    start_date = event_date - pd.DateOffset(months=offset_months)
    end_date = event_date + pd.DateOffset(months=offset_months)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # make it weekly because we know the dataset is not perfectly weekly
    # df = df.resample('W').mean()

    df[event_name] = np.where(df.index >= event_date, 1, 0)
    df["is_sports"] = np.where(df["category"] == "Sports", 1, 0)

    # Create a new column for the interaction term
    df['interaction'] = df[event_name] * df["is_sports"]

    # Check for missing values in the variables
    print("Missing values\n", df[[x_column, 'is_sports',
          event_name, 'interaction']].isnull().sum())

    df = df.dropna(subset=[x_column, 'is_sports', event_name, 'interaction'])

    # Fit the linear regression model
    model = smf.ols(f'{x_column} ~ is_sports + {event_name} + interaction', data=df).fit()

    # Create summary for visualization
    summary = df.groupby([event_name, 'is_sports'])[
        x_column].mean().reset_index()

    summary['Group'] = summary['is_sports'].apply(
        lambda x: 'Sports' if x == 1 else 'Control')

    summary['Period'] = summary[event_name].apply(
        lambda x: 'Post-Treatment' if x == 1 else 'Pre-Treatment')
    

    # Always use orange for sports and skyblue for control
    pallete_plt = {'Sports': 'orange', 'Control': 'skyblue'}
    pallete_plt_fitted = {'Sports_fitted': 'darkorange', 'Control_fitted': 'dodgerblue'}

    # Bar Plot
    sns.barplot(x='Period', y=x_column, hue='Group', data=summary, palette=pallete_plt, hue_order=['Control', 'Sports'])
    plt.title(f'Average {x_column.replace("_", " ").capitalize()} by Group and Period')
    plt.ylabel(f'Average {x_column.replace("_", " ").capitalize()}')
    plt.show()

    # Plots are done with weekly data
    # Time Series Plot
    time_summary = df.groupby([y_column, "is_sports"])[[x_column]].mean().reset_index().copy()
    time_summary.sort_values(by=y_column, inplace=True)
    # Resample the data to weekly frequency
    time_summary_copy = time_summary.copy()
    time_summary_copy.set_index(y_column, inplace=True)
    time_summary_copy.index = pd.to_datetime(time_summary_copy.index)

    # Resample the data to weekly frequency because it isn't perfectly weekly ( we have majority of weekly data, but some are daily and others are every 6 days)
    weekly_sports = time_summary_copy[time_summary_copy['is_sports'] == 1].resample('W').mean()
    weekly_control = time_summary_copy[time_summary_copy['is_sports'] == 0].resample('W').mean()

    # Plot the resampled data
    plt.figure(figsize=(14, 7))
    plt.plot(weekly_sports.index, weekly_sports[x_column], label='Sports', marker='o', color=pallete_plt['Sports'])
    plt.plot(weekly_control.index, weekly_control[x_column], label='Control', marker='o', color=pallete_plt['Control'])
    plt.axvline(event_date, color='red', linestyle='--', label='Treatment Date')
    plt.title(f'{x_column} Over Time by Group (Weekly Resampled)')
    plt.xlabel('Date')
    plt.ylabel(f'Average {x_column.replace("_", " ").capitalize()}')
    plt.legend()
    plt.show()

    # plot model
    def plot_model_results_both_groups():
        df_model_visualize = df.copy()[[x_column]]
        df_model_visualize['fitted'] = model.fittedvalues
        df_model_visualize['residuals'] = model.resid

        df_model_visualize = df_model_visualize.groupby(df_model_visualize.index.date).sum()
        df_model_visualize.index = pd.to_datetime(df_model_visualize.index)
        df_model_visualize = df_model_visualize.resample('W').mean()


        plt.figure(figsize=(12, 6))
        plt.plot(df_model_visualize.index, df_model_visualize[x_column], label='Actual')
        plt.plot(df_model_visualize.index, df_model_visualize['fitted'], label='Fitted', linestyle='--')
        plt.axvline(event_date, color='red', linestyle='--', label='Event Date')
        plt.title(f'Difference in Difference Analysis: {event_name}')
        plt.xlabel('Date')
        plt.ylabel(x_column.replace('_', ' ').capitalize())
        plt.legend()
        plt.show()

    def plot_model_results_group_and_control():
        df_control = df[df['is_sports'] == 0].copy()
        df_sports = df[df['is_sports'] == 1].copy()

        # Do predictions with the model for both groups
        df_control['fitted'] = model.predict(df_control)
        df_sports['fitted'] = model.predict(df_sports)

        df_control = df_control[[x_column, 'fitted']]
        df_sports = df_sports[[x_column, 'fitted']]

        df_control.groupby(df_control.index.date).sum()
        df_control.index = pd.to_datetime(df_control.index)
        df_control = df_control.resample('W').mean()

        df_sports.groupby(df_sports.index.date).sum()
        df_sports.index = pd.to_datetime(df_sports.index)
        df_sports = df_sports.resample('W').mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df_control.index, df_control[x_column], label='Control Actual', color=pallete_plt['Control'])
        plt.plot(df_control.index, df_control['fitted'], label='Control Fitted', linestyle='--', color=pallete_plt_fitted['Control_fitted'])

        plt.plot(df_sports.index, df_sports[x_column], label='Sports Actual', color=pallete_plt['Sports'])
        plt.plot(df_sports.index, df_sports['fitted'], label='Sports Fitted', linestyle='--', color=pallete_plt_fitted['Sports_fitted'])
        plt.axvline(event_date, color='red', linestyle='--', label='Event Date')
        plt.title(f'Difference in Difference Analysis: {event_name}')
        plt.xlabel('Date')
        plt.ylabel(x_column.replace('_', ' ').capitalize())
        plt.legend()
        plt.show()

    # box plot pre vs post
    def box_plot_pre_vs_post():
        df_boxplot = df.copy()
        df_boxplot['Period'] = df_boxplot[event_name].apply(
            lambda x: 'Post-Treatment' if x == 1 else 'Pre-Treatment')
        
        pallete_plt = {0: 'skyblue', 1: 'orange'} # 0 is control, 1 is sports

        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Period", y=x_column, hue='is_sports', data=df_boxplot, palette=pallete_plt)
        plt.title('Boxplot of Delta Views by Group and Period')
        plt.ylabel('Delta Views')
        plt.show()

    # plot the difference in difference bars
    def difference_bars():
        df_diff_visualize = df.copy()[[x_column, 'is_sports']]
        df_diff_visualize['fitted'] = model.fittedvalues

        # difference of pre-event and post-event for control and sports
        control_diff = df_diff_visualize[(df_diff_visualize["is_sports"] == 0) & (df_diff_visualize.index < event_date)][x_column].mean() - df_diff_visualize[(df_diff_visualize["is_sports"] == 0) & (df_diff_visualize.index >= event_date)][x_column].mean()

        sports_diff = df_diff_visualize[(df_diff_visualize["is_sports"] == 1) & (df_diff_visualize.index < event_date)][x_column].mean() - df_diff_visualize[(df_diff_visualize["is_sports"] == 1) & (df_diff_visualize.index >= event_date)][x_column].mean()

        diff_in_diff_values = sports_diff - control_diff

        plt.figure(figsize=(10, 8))
        plt.bar(['Control', 'Sports'], [control_diff, sports_diff], color=[pallete_plt['Control'], pallete_plt['Sports']])
        plt.title(f'Difference in {x_column.capitalize()} Change (Pre vs. Post) "Negative values indicate an increase after the event"')
        plt.ylabel(f'Change in Average {x_column.capitalize()}')
        plt.xlabel('Group')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.show()
    
    plot_model_results_both_groups()
    plot_model_results_group_and_control()
    box_plot_pre_vs_post()
    difference_bars()

    print(model.summary())

    print(f"P-values: {model.pvalues}")

    return model