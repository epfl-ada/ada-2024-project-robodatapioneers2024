
# Game-changer: How do the major sports events influence YouTube engagement?

## Abstract
Major sports events like the World Cup, Olympics, and NBA are global spectacles that captivate billions and ignite conversations worldwide. Beyond the thrill of the games, these events significantly reshape digital landscapes, particularly on platforms like YouTube, which, with its vast global reach and accessibility, serves as a central hub where anyone, anywhere, can engage with and amplify the excitement of these moments. Our motivation is to understand the dynamic interplay between major sports events and online user engagement. \
This project explores how such important sports happenings influence YouTube engagement and content trends. Then, we will identify which types of sports videos become viral and attract people, how engagement levels fluctuate, and whether these events elevate the visibility of minority sports. Ultimately, our findings could offer valuable strategies to enhance audience engagement, foster community connections, and potentially leverage these events to promote healthier lifestyles and social well-being.





## Research Questions
* How does YouTube's sports content change over time, particularly before, during, and after major sports events?
* How do different major sports events (e.g., World Cup, Olympics, NBA Finals) uniquely impact YouTube engagement, trends, and general interests in sports content? What types of sports-related content tend to go viral during major sports events?
* Which sports gain increased visibility during major events? Do minority sports benefit in particular?
* Do major sports events attract users who are non-typical sports audiences on YouTube? How do these events influence their engagement levels and long-term interest in sports content?


## Proposed additional datasets
We will use only YouNiverse dataset.

## Methods
#### Part 0 - Data Preparation and Exploration
Due to the YouNiverse dataset's vast size, processing all data was impractical. Focusing on sports-related YouTube content, we sampled the data and built master datasets by cleaning and categorizing the original information. We extracted sports data from the 'time series,' 'channel,' 'metadata,' and 'comment' categories using a systematic approach, highlighting the critical importance of our initial data processing. 

Approach:
* Cleaning data
  * Remove unnecessary columns and appropriately handle missing values by thoroughly examining the dataset contents.
* Exploring the data
  * Gain a comprehensive overview of the dataset by analyzing key statistics and examining the distribution of various sports and also all different categories of channels in the dataset
* Categorizing data (we implement this before each part as necessary)
  * Find the sport and specific event(Olympic, World Cup, and NBA) related data
  * Use NLP models to analyze video titles and descriptions to identify the video content (whether they are highlights/live streams of some matches, the ceremony, funny moments…)

#### Part 1 - Overview of trends in sports-related content
To address our first research question, we analyzed sports-related content on YouTube to identify major events and influential sports within the platform's communities. This foundational analysis is essential for understanding the dynamics that shape YouTube's sports content and sets the stage for deeper investigations in subsequent phases. 

Approach
* Trend Visualization with Word Clouds: Generate word clouds to visualize the most frequent terms in sports-related videos, highlighting prevailing topics and interests.
* Time Series Analysis: Plot the number of sports videos and channels uploaded over time to observe growth patterns and seasonal trends.
* Comparative Analysis: Compare the volume of sports-related content against other categories to assess its relative popularity and influence within the YouTube community.

#### Part 2 - Major sports events comparison
We will focus on major sports events like the World Cup, Olympics, and NBA to delve deeper into their impact on YouTube engagement trends and interests. For each event, we'll analyze the changes in user engagement metrics and content volume over time to determine how each one influences YouTube communities. Additionally, we'll identify which types of sports content go viral during these events by classifying and analyzing the popularity of different content categories (e.g., highlights, tutorials, fan reactions). 

Approach
* Engagement metrics Analysis
  * Identifying key engagement metrics—likes, dislikes, view counts, comments, and subscriber growth—from the dataset. By comparing these metrics across major sports events like the World Cup, Olympics, and NBA, we can assess each event's impact on user interactions and content popularity on YouTube.
* Sentiment analysis
  * Implement sentiment analysis for each event and compare between events
Analyze whether the titles of videos have positive or negative sentiments to evaluate the creator’s sentiments. This will enhance the depth of engagement analysis by providing a nuanced view of public opinion, emotional engagement, and potential drivers behind content popularity.
* Statistical Testing
  * Significant Changes Identification: Perform statistical tests to detect significant variations in engagement metrics during and after major events.
  * Correlation Analysis: Explore the relationship between sports events and increases in views and subscribers to determine the extent of their impact on YouTube growth.
 
#### Part 3 - Sport-based analysis
This section focuses on each sport of interest over major events. We use statistical tests to reveal how these events impact each sport and explore the correlations between different sports. This analysis may provide a potential strategy for minority sports to gain attention. \
Our approach involves applying the same methodologies as in Part 2, with a primary focus on comparing user engagement between major and minor sports.

#### Part 4 - User interests transition
We focus on how users’ interest in sports is influenced by major sporting events. Specifically, we aim to identify frequent YouTube commenters and track their activity before, during, and after these events. This analysis helps to understand whether major sports events encourage users to explore new sports and how long these influences persist. For instance, did dedicated baseball fans show interest in other sports and start engaging with their content (i.e., by commenting on videos) during and after the Olympics?

Approach
* User clustering
  *   Classify videos based on user comments and count them by category or sports. Represent users as vectors, with each element indicating their comment count per video category. Cluster users (e.g., frequent baseball viewers, rare sports watchers) and analyze changes in cluster sizes and commenting behavior over time.


## Proposed timeline
We did some basic analysis for each part in Milestone 2 to make sure of the feasibility of our plan, but we will do it in more detail in Milestone 3. \
15/11 - 22/11: Part 1, 2 \
23/11 - 29/11: Part 2 & 3 \
30/11 - 06/12: Part 4 \
06/12 - 13/12: Wrapping up and sophisticate outcomes, and write a story \
13/12 - 20/12: Cleaning the repository, wrapping up the data story webpage 


## Organization within the team
Andres: Major sports events comparison(part2) \
Huyen: Sentiment analysis(part2,3) \
Keisuke: User clustering(part 4) \
Yugo: Sport-based analysis(part3) \
Zahra: Overview of trends in sport-related(part 1)








