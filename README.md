# Natural Phenomena Detection from Tweets  
The project aims to develop a system that detects and categorizes natural disaster related tweets.
This system is designed to assist emergency services, humanitarian organizations, and news
agencies in responding to critical events such as wildfires, floods, hurricanes, tornadoes, and
droughts. This tool is also useful for researchers studying social media trends during natural
disasters. Users will interact with a dashboard that displays categorized tweets. Additionally, it
also alerts users to stay updated about the activity of the disasters.

Key Actions and Features:  
- Tweet Collection: The system monitors for new tweets related to specific disaster
keywords.
- Categorization: Each tweet is classified into one of the categories like wildfire, flood,
hurricane, tornodo, or drought using machine learning alogorithms.
- Dashboard: Users are provided with a dashboard that displays categorized tweets.
- Alerts: In case of severe activity related to a particular disaster, the system sends alerts to
users.

Workflow:  
1.	Data Ingestion: Collecting disaster related tweets for training
2.	Preprocessing: Text preprocessing including tokenization, removal of irrelevant data
3.	Feature Extraction: The data is converted into feature vectors
4.	Classification: A naïve bayes classifier is used to categorize each tweet
5.	Dashboard Display: The tweets are displayed on a dashboard
6.	Send Alerts: Alerts users based on the volume of specific disaster tweets

Project Design:   
Programming Language: Python  
IDE: Visual Studio Code, Google Colab  
Libraries/Packages:  
•	Scikit-learn for machine learning and classification  
•	NLTK for natural language processing, tokenization  
•	Flask for building the dashboard  
•	Pandas for data manipulation  
•	Matplotlib for visualizing results  

Live at: https://disaster-ir-app.onrender.com/
