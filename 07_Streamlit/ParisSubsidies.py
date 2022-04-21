# STREAMLIT APP
import streamlit as st
st.set_page_config()

# LIBRAIRIES IMPORT
# Data processing
import pandas as pd
import numpy as np

# Data visualisation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# FUNCTIONS
def safe_num(num):
    if isinstance(num, str):
        num = float(num)
    return float('{:.3g}'.format(abs(num)))

def format_number(num):
    num = safe_num(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

# DATA IMPORT
@st.cache
def data_import():
    data = pd.read_feather('../00_DataFiles/05_NLPScoring/ParisSubsidies_NLPScoring.feather')
    return data

data = data_import()

# TITLE
st.title('Paris Subsidies', anchor = 'title')
#st.sidebar.markdown("[Top](#title)", unsafe_allow_html=True)

st.markdown("""---""")

# BUSINESS CASE
st.header('Business case', anchor = 'business_case')
st.markdown('''
***Who*** is benefiting from Paris City Council's subsidies? ***What*** causes is Paris City Council likely to support by granting financial aid?
***When*** are these subsidies voted? ***Where*** do they go? ***Why*** does a non-profit organisation receive a subsididy while another one does not?

The answers we aim at providing to these questions should be of interest to:
- Parisian citizens and taxpayers who want to understand where they taxmoney goes
- Journalists working on Paris' politics and non-profit ecosystem
- Associations that would like to get a subsidy from Paris City Council
''')

st.markdown("""---""")

# DATASET
st.header('Dataset', anchor = 'dataset')

# Sources
st.subheader('Sources')
st.markdown('''
To form our dataset, we have combined data from four sources:
- [Open Data Paris](https://opendata.paris.fr) where we downloaded subsidies requests
- [SIRENE V3 consolidée](https://public.opendatasoft.com/explore/dataset/economicref-france-sirene-v3/) where we extracted legal and statistical data through custom API calls
- [sirene.fr](https://www.sirene.fr/sirene/public/accueil) that we used for missing legal and statistical data imputation and correction
- [Nominatim](https://nominatim.org) for missing geocoordinates (API and manual)
''')

# ...under the hood...
with st.expander('Under the hood'):
    st.markdown('''
    The SIRENE V3 consolidée enriches INSEE's SIRENE data with labels, hierarchies, geocoordinates, etc.
    It also has the huge benefit of having non-restrictive API usage (as long as it remains reasonable) while INSEE's SIRENE soon requires you to pay.
    ''')

# Statistics
@st.cache
def dataset_statistics():
    dataset_size = data.shape[0]
    nb_associations = data['siret'].nunique()
    period = str(data['annee_budgetaire'].min()) + ' - ' + str(data['annee_budgetaire'].max())
    return dataset_size, nb_associations, period

dataset_size, nb_associations, period = dataset_statistics()

st.subheader('Statistics')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label = 'Nb of records', value = f'{dataset_size:,}')
with col2:
    st.metric(label = 'Nb of associations', value = f'{nb_associations:,}')
with col3:
    st.metric(label = 'Analysed period', value = period)

# ...under the hood...
with st.expander('Under the hood'):
    st.markdown('''
    The original number of records present in the Open Data Paris dataset was 80,084.

    Several records had missing or incorrect data (siret, name, etc.) preventing us from joining them with legal and statistical data.
    
    Thanks to manual imputation and data correction on more than 300 records, we eventuall only had to drop 123 records, i.e. less than 0.2% of the dataset.
    
    Could we have save time by dropping the other 300+ records? Yes, but we would have lost some valuable information about specific segments of our dataset: for instance, most of the incorrect records were related to amateur theater troupes.

    Our cleaned and enriched data is under [MIT license](https://raw.githubusercontent.com/eric-martinet/ParisSubsidies/main/LICENSE) so feel free to reuse it.
    Avaialble formats in our [GitHub repo](https://github.com/eric-martinet/ParisSubsidies) are CSV (for universality) and Feather (for performance).
    ''')

st.markdown("""---""")

# EDA
st.header('Exploratory Data Analysis', anchor = 'eda')
st.markdown('It is time to dig into our data!')

# Key facts
def key_facts():
    nb_years = nb_years = data['annee_budgetaire'].nunique()
    avg_yearly_subsidies = int(data.montant_vote.sum()/nb_years)
    avg_yearly_nb_requests = int(dataset_size/nb_years)
    avg_reject_rate = len(data.loc[data.subsidy_granted_bool == False]) / dataset_size
    max_subsidy = data.loc[data.subsidy_granted_bool == True].montant_vote.max()
    min_project_subsidy = data.loc[(data.subsidy_granted_bool == True) & (data.nature_subvention == 'Projet')].montant_vote.min()
    return nb_years, avg_yearly_subsidies, avg_yearly_nb_requests, avg_reject_rate, max_subsidy, min_project_subsidy

nb_years, avg_yearly_subsidies, avg_yearly_nb_requests, avg_reject_rate, max_subsidy, min_project_subsidy = key_facts()

st.subheader('Key facts')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label = 'Avg yearly subsidies', value = format_number(avg_yearly_subsidies)+'€')
    st.caption('This is about 3% of Paris\' yearly budget (1OB€).')
with col2:
    st.metric(label = 'Avg yearly nb of requests', value = f'{avg_yearly_nb_requests:,}')
    st.caption('i.e. one every hour, including the night!')
with col3:
    st.metric(label = 'Avg reject rate', value = f'{avg_reject_rate:.1%}')
    st.caption('Almost 4 out of 10 requests are rejected: what is the pattern?')

# col1, col2 = st.columns(2)
# with col1:
#     st.metric(label = 'Max subsidy', value = format_number(max_subsidy)+'€')
#     st.caption('Granted to "Théâtre Musical de Paris" in 2013 as a "Subvention d\'équilibre".')
# with col2:
#     st.metric(label = 'Min subsidy', value = format_number(min_project_subsidy)+'€')
#     st.caption('Granted to "Assocation Osadhi" in 2019 for the "Printemps des Cimetières".')

# ...under the hood...
with st.expander('Under the hood'):
    st.markdown('''
    In total, it is more than 2.4B€ worth of subsidies that we have analysed over 2013-2021, granted to about 6,800 different associations.
    ''')


# Analysis by Budget year
st.subheader('Analysis by Budget year')
st.write('''The Budget year is the reference period for public accounting. 
By law, each spending must be approved by Paris City Council and allocated to a specific Budget year.
''')

# Total subsidies & Mean subsidy by Budget year
@st.cache
def total_mean_subsidy_by_year():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('annee_budgetaire').agg(total_subsidies = ('montant_vote', 'sum'), mean_subsidy = ('montant_vote', 'mean'))

    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.total_subsidies, name='Total subsidies'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=gb.index, y=gb.mean_subsidy, name='Mean subsidy'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Total subsidies & Mean subsidy by Budget year (in EUR)</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Budget year</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Total</b> subsidies', color = '#636EFA', hoverformat = ',.0f', secondary_y=False)
    fig.update_yaxes(title_text='<b>Mean</b> subsidy', color='#EF553B', showgrid = False, hoverformat = ',.0f', secondary_y=True)

    return fig

fig = total_mean_subsidy_by_year()
st.plotly_chart(fig)
st.caption('''
Till 2020, the total amount of subsidies steadily increased while the mean subsidy decreased: Paris City council funded more and more associations.

2021 saw a drop in both metrics: we can assume that it reflects the financial crisis induced by the covid-19 crisis.
''')

# Number of requests & Reject rate by Budget year

@st.cache
def nb_requests_reject_rate_by_year():

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('annee_budgetaire').agg(nb_dossiers = ('numero_dossier', 'count'))
    ct = pd.crosstab(data.annee_budgetaire, data.subsidy_granted_bool, normalize='index').reindex(gb.index)

    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.nb_dossiers, name='Number of requests'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=ct.index, y=ct[False], name='Reject rate'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Number of requests & Reject rate by Budget year</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Budget year</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Number</b> of requests', color = '#636EFA', hoverformat = ',.0f', secondary_y=False)
    fig.update_yaxes(title_text='<b>Reject</b> rate', color='#EF553B', showgrid = False, tickformat = '.1%', secondary_y=True)

    return fig

fig = nb_requests_reject_rate_by_year()
st.plotly_chart(fig)
st.caption('''
Paris City Council was less and less selective from 2015 to 2018, and still below its medium-term average (39%) in 2019 and 2020.

It never received so many requests as in 2021, but the reject rate soared: we can assume this is again the effect of the financial crisis induced by the covid-19 crisis.
''')

# Analysis by Direction
st.subheader('Analysis by Service')
st.write('''
Paris City has 27 different services or 'directions' (DAC: cultural affairs, DLH: housing, etc.) that manage and process subsidy granting.
''')

# Subsidies by Direction
@st.cache
def subsidies_by_direction():
    #Dataframes
    gb = data.groupby('direction').agg(total_subsidies = ('montant_vote', 'sum')).sort_values(by='total_subsidies', ascending = False)
    gb['avg_yearly_subsidies'] = gb.total_subsidies / nb_years
    gb['avg_yearly_subsidies'] = gb['avg_yearly_subsidies'].astype(int)
    gb = gb.drop('total_subsidies', axis = 1)

    # Group small
    gb_small = {'Others':0}
    lst_small = []
    threshold = 0.05 # 5% of total

    for dir in gb.index:
        if gb.loc[dir, 'avg_yearly_subsidies'] < gb['avg_yearly_subsidies'].sum() * threshold:
            gb_small['Others'] += gb.loc[dir, 'avg_yearly_subsidies']
            lst_small.append(dir)

    gb=pd.concat([gb.drop(lst_small, axis = 0), pd.DataFrame.from_dict(gb_small, orient='index', columns=['avg_yearly_subsidies'])])


    # Pie chart
    fig = go.Figure(data=[go.Pie(labels=gb.index, values=gb.avg_yearly_subsidies, textinfo='label+percent', sort = False, pull=[0,0,0,0,0,0.1])])

    # Layout
    fig.update_layout(
        title_text='<b>Avg share in yearly subsidies by Direction</b>',
        showlegend = False
    )

    return fig

fig = subsidies_by_direction()
st.plotly_chart(fig)
st.caption('''
The top 2 Directions (DAC = Direction des Affaires Culturelles, DFPE = Direction des Familles et de la Petite Enfance) account for more than 50% of the granted subsidies.

Adding DAE (Direction de l'Attractivité et de l'Emploi), DASES (Direction de l'Action Sociale de l'Enfance et de la Santé), and DRH (Direction des Ressources Humaines), we get to more than 80%.
''')