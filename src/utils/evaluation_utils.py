import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def diff_in_diff(df_orig, y_column, x_column, event_date, event_name, sport_name, offset_months=4):
    # Column for event (0 before the event, 1 after the event)
    df = df_orig[[y_column, x_column,  "sport_name"]].copy()
    print(df.shape)
    df[y_column] = pd.to_datetime(df[y_column])
    df.set_index(y_column, inplace=True)
    print(df.shape)
    df.index = pd.to_datetime(df.index)
    print(df.index)

    # df = df.groupby(df.index.date).sum()
    # is is correct to make a sum here, or would it be better to do a mean?
    df.sort_index(inplace=True)
    print(df.shape)
    df.index = pd.to_datetime(df.index)
    print(df.shape)

    start_date = event_date - pd.DateOffset(months=offset_months)
    end_date = event_date + pd.DateOffset(months=offset_months)
    df = df[(df.index >= pd.to_datetime(start_date))
            & (df.index <= pd.to_datetime(end_date))]
    print(df.shape)
    # make it weekly because we know the dataset is not perfectly weekly
    # df = df.resample('W').mean()

    df[event_name] = np.where(df.index >= event_date, 1, 0)
    df["treatment_sport"] = np.where(df["sport_name"] == sport_name, 1, 0)
    print(df.shape)
    # Create a new column for the interaction term
    df['interaction'] = df[event_name] * df["treatment_sport"]
    # Check for missing values in the variables
    print("Missing values\n", df[[x_column, 'treatment_sport',
          event_name, 'interaction']].isnull().sum())

    df = df.dropna(subset=[x_column, 'treatment_sport',
                   event_name, 'interaction'])

    # Fit the linear regression model
    model = smf.ols(
        f'{x_column} ~ treatment_sport + {event_name} + interaction', data=df).fit()

    # Create summary for visualization
    summary = df.groupby([event_name, 'treatment_sport'])[
        x_column].mean().reset_index()

    summary['Group'] = summary['treatment_sport'].apply(
        lambda x: 'Sport' if x == 1 else 'Control')

    summary['Period'] = summary[event_name].apply(
        lambda x: 'Post-Treatment' if x == 1 else 'Pre-Treatment')

    # Always use orange for sports and skyblue for control
    pallete_plt = {'Sport': 'orange', 'Control': 'skyblue'}
    pallete_plt_fitted = {'Sports_fitted': 'darkorange',
                          'Control_fitted': 'dodgerblue'}

    # Bar Plot
    sns.barplot(x='Period', y=x_column, hue='Group', data=summary,
                palette=pallete_plt, hue_order=['Control', 'Sport'])
    plt.title(
        f'Average {x_column.replace("_", " ").capitalize()} by Group and Period')
    plt.ylabel(f'Average {x_column.replace("_", " ").capitalize()}')
    plt.show()

    # Plots are done with weekly data
    # Time Series Plot
    time_summary = df.groupby([y_column, "treatment_sport"])[
        [x_column]].mean().reset_index().copy()
    time_summary.sort_values(by=y_column, inplace=True)
    # Resample the data to weekly frequency
    time_summary_copy = time_summary.copy()
    time_summary_copy.set_index(y_column, inplace=True)
    time_summary_copy.index = pd.to_datetime(time_summary_copy.index)

    # Resample the data to weekly frequency because it isn't perfectly weekly ( we have majority of weekly data, but some are daily and others are every 6 days)
    weekly_sports = time_summary_copy[time_summary_copy['treatment_sport'] == 1].resample(
        'W').mean()
    weekly_control = time_summary_copy[time_summary_copy['treatment_sport'] == 0].resample(
        'W').mean()

    # Plot the resampled data
    plt.figure(figsize=(14, 7))
    plt.plot(weekly_sports.index, weekly_sports[x_column],
             label=sport_name, marker='o', color=pallete_plt['Sport'])
    plt.plot(weekly_control.index, weekly_control[x_column],
             label='Control', marker='o', color=pallete_plt['Control'])
    plt.axvline(event_date, color='red',
                linestyle='--', label='Treatment Date')
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

        df_model_visualize = df_model_visualize.groupby(
            df_model_visualize.index.date).sum()
        df_model_visualize.index = pd.to_datetime(df_model_visualize.index)
        df_model_visualize = df_model_visualize.resample('W').mean()

        colors_pastel = ["#558a6a", "#97bd88", "#e8e791", "#e0878a"]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df_model_visualize.index, df_model_visualize[x_column],
                 label='Actual', color=colors_pastel[0])  # First color for actual data
        plt.plot(df_model_visualize.index, df_model_visualize['fitted'],
                 label='Fitted', linestyle='--', color=colors_pastel[1])  # Second color for fitted data
        plt.axvline(event_date, color=colors_pastel[3],
                    linestyle='--', label='Event Date')  # Fourth color for event line
        plt.title(f'Difference in Difference Analysis: {event_name}')
        plt.xlabel('Date')
        plt.ylabel(x_column.replace('_', ' ').capitalize())
        plt.legend()
        plt.show()

    def plot_model_results_group_and_control():
        df_control = df[df['treatment_sport'] == 0].copy()
        df_sports = df[df['treatment_sport'] == 1].copy()

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
        colors_pastel = ["#558a6a", "#97bd88", "#e8e791", "#e0878a"]
        # Control Actual
        plt.plot(df_control.index, df_control[x_column],
                 label='Control Actual', color=colors_pastel[0])

        # Control Fitted
        plt.plot(df_control.index, df_control['fitted'],
                 label='Control Fitted', linestyle='--', color=colors_pastel[1])

        # Sports Actual
        plt.plot(df_sports.index, df_sports[x_column],
                 label='Sports Actual', color="#e50000")

        # Sports Fitted
        plt.plot(df_sports.index, df_sports['fitted'],
                 label='Sports Fitted', linestyle='--', color="#91060d")

        # Event Date
        plt.axvline(event_date, color="#f2d200",  # Red pastel from your earlier palette
                    linestyle='--', label='Event Date')

        # Titles and Labels
        plt.title(f'Difference in Difference Analysis: {event_name}')
        plt.xlabel('Date')
        plt.ylabel(x_column.replace('_', ' ').capitalize())

        # Legend
        plt.legend()

        # Show plot
        plt.show()

    # box plot pre vs post

    def box_plot_pre_vs_post():
        df_boxplot = df.copy()
        df_boxplot['Period'] = df_boxplot[event_name].apply(
            lambda x: 'Post-Treatment' if x == 1 else 'Pre-Treatment')

        pallete_plt = {0: 'skyblue', 1: 'orange'}  # 0 is control, 1 is sports

        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Period", y=x_column, hue='treatment_sport',
                    data=df_boxplot, palette=pallete_plt)
        plt.title('Boxplot of Delta Views by Group and Period')
        plt.ylabel('Delta Views')
        plt.show()

    # plot the difference in difference bars
    def difference_bars():
        df_diff_visualize = df.copy()[[x_column, 'treatment_sport']]
        df_diff_visualize['fitted'] = model.fittedvalues

        # difference of pre-event and post-event for control and sports
        control_diff = df_diff_visualize[(df_diff_visualize["treatment_sport"] == 0) & (df_diff_visualize.index < event_date)][x_column].mean(
        ) - df_diff_visualize[(df_diff_visualize["treatment_sport"] == 0) & (df_diff_visualize.index >= event_date)][x_column].mean()

        sports_diff = df_diff_visualize[(df_diff_visualize["treatment_sport"] == 1) & (df_diff_visualize.index < event_date)][x_column].mean(
        ) - df_diff_visualize[(df_diff_visualize["treatment_sport"] == 1) & (df_diff_visualize.index >= event_date)][x_column].mean()

        diff_in_diff_values = sports_diff - control_diff

        plt.figure(figsize=(10, 8))
        plt.bar(['Control', sport_name], [control_diff, sports_diff],
                color=[pallete_plt['Control'], pallete_plt['Sport']])
        plt.title(
            f'Difference in {x_column.capitalize()} Change (Pre vs. Post) "Negative values indicate an increase after the event"')
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
