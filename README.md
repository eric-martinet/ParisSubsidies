<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Paris Subsidies (Subventions de la Mairie de Paris)
*Eric Martinet*

**Data Analytics bootcamp @IronHack Paris, Feb-Apr 22**

## Project definition

This is my final project for the Data Analytics bootcamp I did in Feb-Apr 2022 at [IronHack](https://www.ironhack.com/fr/data-analytics/paris) in Paris.

Using public open data only, it aims at visualising and analysing the subsidies granted by Paris City Council to non-profit organisations, following the [Five Ws principle](https://en.wikipedia.org/wiki/Five_Ws).

The completion of this project has taken 4 days before being presented to a jury.

## Business case
*Who* is distribution Paris City Council's subsidies? *What* fields and activities is Paris City Council likely to support? *When* are these subsidies granted? *Where* do they go? *Why* does a non-profit organisation receive a subsididy while another one does not?

The answers we aim at providing to these questions should be of interest to:

- Parisian citizens and taxpayers who want to understand where they taxmoney goes
- Journalists working on Parisian politics and non-profit ecosystem
- Non-profit organisations that would like to get a subsidy from Paris City Council

## Plan
To answer the above questions, we need to:

1. Collect the data using open data sources including [Open Data Paris](https://opendata.paris.fr), [sirene.fr](https://www.sirene.fr/sirene/public/accueil), [OpenDataSoft](https://public.opendatasoft.com/explore/dataset/economicref-france-sirene-v3/).
2. Clean the data.
3. Explore the data to define the relevant metrics, analyses, and projections (EDA): we might need to iterate several times on data cleaning and EDA.
4. Define the models & algorithms that best describe and explain the subsidies granted by Paris City Council, taking into account quantitative (amount, etc.) & qualitative data (project nature and description, etc.). Tools to be used: statistics, probabilies, natural language processing (NLP).
5. Synthetise insights and communicate them to the public through a [Streamlit](https://streamlit.io) app.

## Results
Please find the main results on our [Streamlit app: Paris subsidies](https://share.streamlit.io/eric-martinet/parissubsidies/main/07_Streamlit/ParisSubsidies.py)


## Additional links
- [Original repo](https://github.com/eric-martinet/ParisSubsidies)
- [Official platform to request a subsidy from Paris City Council](https://www.paris.fr/pages/les-demandes-de-subventions-5334)



