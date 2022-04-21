# STREAMLIT APP
import joblib
from scipy.fft import dct
import streamlit as st
st.set_page_config()

# LIBRAIRIES IMPORT
# Data processing
import pandas as pd
import numpy as np

# Data visualisation
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP
import spacy
import string
from unidecode import unidecode

# ML loading
from joblib import dump, load

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

def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    text2 = unidecode(text2)
    
    return text2.lower()

# DATA IMPORT
@st.cache
def data_import():
    data = pd.read_feather('../00_DataFiles/05_NLPScoring/ParisSubsidies_NLPScoring.feather')
    return data

data = data_import()

# SIDEBAR
st.sidebar.markdown('''
    ### NAVIGATION

    [Back to top](#title)

    [Business case](#business_case)

    [Dataset](#dataset)
    - [Sources](#dataset-sources)
    - [Statistics](#dataset-statistics)

    [EDA](#eda)
    - [Key facts](#eda-key_facts)
    - [Distribution analysis](#eda-distribution)
    - [Analysis by Budget year](#eda-budget_year)
    - [Analysis by Direction](#eda-direction)
    - [Geo-analysis](#eda-geo)

    [Machine learning: request success](#machine_learning)
    - [Pipeline](#ml-pipeline)
    - [First try](#ml-1)
    - [Reflecting on first try](#ml-reflecting)
    - [Second try](#ml-2)
    - [Test your request!](#ml-test_request)

    [Machine learning: predict amount](#ml_amount)

    [Fun fact](#fun_facts)

''', unsafe_allow_html=True)

# TITLE
st.title('Paris Subsidies', anchor = 'title')

st.markdown('''
    *by Eric Martinet, 22 April 2022*

    This is my final project for the Data Analytics bootcamp I did in Feb-Apr 2022 at [IronHack](https://www.ironhack.com/fr/data-analytics/paris) in Paris.

    Using public open data only, it aims at visualising and analysing the subsidies granted by Paris City Council to non-profit organisations, following the [Five Ws principle](https://en.wikipedia.org/wiki/Five_Ws).

    You can find all datafiles, Jupyter notebooks and python scripts as well as the source code of this app on our [GitHub repo](https://github.com/eric-martinet/ParisSubsidies).
''')

st.markdown("""---""")

# BUSINESS CASE
st.header('Business case', anchor = 'business_case')

st.markdown('''
***Who*** is granting Paris City Council's subsidies? ***What*** causes is Paris City Council likely to support?
***When*** are these subsidies granted? ***Where*** do they go? ***Why*** does a non-profit organisation receive a subsididy while another one does not?

The answers we aim at providing to these questions should be of interest to:
- Parisian citizens and taxpayers who want to understand where they taxmoney goes
- Journalists working on Paris' politics and non-profit ecosystem
- Associations that would like to get a subsidy from Paris City Council
''')

st.markdown("""---""")

# DATASET
st.header('Dataset', anchor = 'dataset')

# Sources
st.subheader('Sources', anchor='dataset-sources')

st.markdown('''
To form our dataset, we have combined data from four sources:
- [Open Data Paris](https://opendata.paris.fr) where we downloaded subsidies requests
- [SIRENE V3 consolidée](https://public.opendatasoft.com/explore/dataset/economicref-france-sirene-v3/) where we extracted legal and statistical data through custom API calls
- [sirene.fr](https://www.sirene.fr/sirene/public/accueil) that we used for missing legal and statistical data imputation and correction
- [Nominatim](https://nominatim.org) for missing geocoordinates (API and manual)
''')

# ...under the hood...
with st.expander('Under the hood...'):
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

st.subheader('Statistics', anchor='dataset-statistics')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label = 'Nb of records', value = f'{dataset_size:,}')
with col2:
    st.metric(label = 'Nb of associations', value = f'{nb_associations:,}')
with col3:
    st.metric(label = 'Analysed period', value = period)

# ...under the hood...
with st.expander('Under the hood...'):
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

st.subheader('Key facts', anchor='eda-key_facts')

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


# ...under the hood...
with st.expander('Under the hood...'):
    st.markdown('''
    In total, it is more than 2.4B€ worth of subsidies that we have analysed over 2013-2021, granted to about 6,800 different associations.
    ''')
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label = 'Max subsidy', value = format_number(max_subsidy)+'€')
        st.caption('Granted to "Théâtre Musical de Paris" in 2013 as a "Subvention d\'équilibre".')
    with col2:
        st.metric(label = 'Min subsidy', value = format_number(min_project_subsidy)+'€')
        st.caption('Granted to "Assocation Osadhi" in 2019 for the "Printemps des Cimetières".')


# Distribution of subsidies
st.subheader('Distribution analysis', anchor='eda-distribution')
st.write('''The Budget year is the reference period for public accounting. 
By law, each spending must be approved by Paris City Council and allocated to a specific Budget year.
''')

@st.cache
def distribution_analysis():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('montant_vote_cat').agg(nb_dossiers = ('numero_dossier', 'count'), total_subsidies = ('montant_vote', 'sum'))
    gb['avg_yearly_subsidies'] = gb.total_subsidies / nb_years
    gb['avg_nb_dossiers'] = gb.nb_dossiers / nb_years

    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.avg_yearly_subsidies, name='Yearly subsidies'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=gb.index, y=gb.avg_nb_dossiers, name='Number of requests'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Yearly subsidies and Number of Requests by Subsidy size</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Subsidy</b> size')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Yearly</b> subsidies', color = '#636EFA', secondary_y=False)
    fig.update_yaxes(title_text='<b>Number</b> of requests', color='#EF553B', showgrid = False, tickformat = '.0f', secondary_y=True)

    return fig

fig = distribution_analysis()
st.plotly_chart(fig)
st.caption('''
Paris City Council essentially grants subsidies in the 1-10k EUR range, but also process many between 10 and 100k EUR.
However, in terms of total amounts, most of the money goes to project above 100k EUR and even 1M EUR.
''')

# ...under the hood...
with st.expander('Under the hood...'):
    st.markdown('''
    Given the range of distributed amounts, we used a log10 scale to group the subsidies by size.
    ''')


# Analysis by Budget year
st.subheader('Analysis by Budget year', 'eda-budget_year')
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
        title_text='<b>Total subsidies & Mean subsidy by Budget year</b>',
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
st.subheader('Analysis by Direction', anchor='eda-direction')
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
        title_text='<b>Avg share in Yearly Subsidies by Direction</b>',
        showlegend = False
    )

    return fig


fig = subsidies_by_direction()
st.plotly_chart(fig)
st.caption('''
The top 2 Directions (DAC = Direction des Affaires Culturelles, DFPE = Direction des Familles et de la Petite Enfance) account for more than 50% of the granted subsidies.

Adding the DAE (Direction de l'Attractivité et de l'Emploi), the DASES (Direction de l'Action Sociale de l'Enfance et de la Santé), and the DRH (Direction des Ressources Humaines), we get to more than 80% of the total.
''')

# Requests by Direction
@st.cache
def nb_requests_by_direction():
    #Dataframes
    gb = data.groupby('direction').agg(nb_requests = ('numero_dossier', 'count')).sort_values(by='nb_requests', ascending = False)
    gb['avg_yearly_nb_requests'] = gb.nb_requests / nb_years
    gb['avg_yearly_nb_requests'] = gb['avg_yearly_nb_requests'].astype(int)
    gb = gb.drop('nb_requests', axis = 1)

    # Group small
    dir_ind = ['DAC', 'DFPE', 'DAE', 'DASES', 'DRH']
    gb_others = {'Others':0}
    lst_others = []

    for dir in gb.index:
        if dir not in dir_ind:
            gb_others['Others'] += gb.loc[dir, 'avg_yearly_nb_requests']
            lst_others.append(dir)

    gb=pd.concat([gb.drop(lst_others, axis = 0), pd.DataFrame.from_dict(gb_others, orient='index', columns=['avg_yearly_nb_requests'])])
    
    # Sort as previous chart
    rule = {
        'DAC': 1,
        'DFPE': 2,
        'DAE': 3,
        'DASES': 4,
        'DRH':5,
    }
    gb.reset_index(level = 0, drop = False, inplace=True)
    gb.sort_values(by='index', key=lambda x: x.apply(lambda y: rule.get(y, 1000)), inplace = True)
    gb.set_index('index', inplace = True)

    # Pie chart
    fig = go.Figure(data=[go.Pie(labels=gb.index, values=gb.avg_yearly_nb_requests, textinfo='label+percent', sort = False, pull=[0,0,0,0,0,0.1])])

    # Layout
    fig.update_layout(
        title_text='<b>Avg share in Yearly Requests by Direction</b>',
        showlegend = False,

    )

    return fig

fig = nb_requests_by_direction()
st.plotly_chart(fig)
st.caption('''
However these 5 directions do not account for even half of the requests!

For instance, the DRH only handles subsidies for Paris City's employees two representative bodies (Action Sociale des Personnels de la Ville de Paris with 4 requests per year, and AGOSPAP - APHP with 2 requests per year).
''')

@st.cache
def nb_requests_reject_rate_by_direction():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('direction').agg(avg_nb_requests = ('numero_dossier', 'count')).sort_values(by='avg_nb_requests', ascending = False)
    gb.avg_nb_requests = np.ceil(gb.avg_nb_requests / nb_years)
    ct = pd.crosstab(data.direction, data.subsidy_granted_bool, normalize='index').reindex(gb.index)


    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.avg_nb_requests, name='Number of requests'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=ct.index, y=ct[False], name='Reject rate'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Number of requests (yearly average) & Reject rate by Direction</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Direction</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Number</b> of requests <i>(by year)</i>', color = '#636EFA', secondary_y=False)
    fig.update_yaxes(title_text='<b>Reject</b> rate <i>(average)</i>', color='#EF553B', showgrid = False, range=[0,0.8],tickformat = '.0%', secondary_y=True)

    return fig

fig = nb_requests_reject_rate_by_direction()
st.plotly_chart(fig)
st.caption('''
The DDCT (Direction de la Démocratie, des Citoyen.ne.s et des Territoires) is the Direction that processes the most requests. They distribute about 10M€ per year.

Another important Direction we have not seen before is the DJS (Direction de la Jeunesse et des Sports). They distribute about 11M€ per year.
''')

# Geo analysis
st.subheader('Geo analysis', anchor='eda-geo')
st.write('''
Is Paris City Council essentially subsidising Parisian associations?
''')

@st.cache
def geo_paris_idf_beyond():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('geo_cat').agg(total_subsidies = ('montant_vote', 'sum')).sort_values(by='total_subsidies', ascending = False)
    gb['avg_yearly_subsidies'] = gb.total_subsidies / nb_years
    ct = pd.crosstab(data.geo_cat, data.subsidy_granted_bool, normalize='index').reindex(gb.index)

    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.avg_yearly_subsidies, name='Yearly subsidies (average)'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=ct.index, y=ct[False], name='Reject rate'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Yearly subsidies and Reject rate by Geography</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Geography</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Yearly</b> subsidies', color = '#636EFA', secondary_y=False)
    fig.update_yaxes(title_text='<b>Reject</b> rate', color='#EF553B', showgrid = False, tickformat = '.1%', secondary_y=True)

    return fig

fig = geo_paris_idf_beyond()
st.plotly_chart(fig)
st.caption('''
96% of the subsidies go to associations headquartered in Paris (however this number includes para-public institutions and big subsidies).

In addition, associations in Île-de-France and beyond are less likely to see their request accepted.
''')



@st.cache
def geo_paris_idf_beyond_1_10k():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    data_red = data.loc[(data.montant_vote_scale == 3)]
    gb = data_red.groupby('geo_cat').agg(total_subsidies = ('montant_vote', 'sum'), nb_requests = ('numero_dossier', 'count')).sort_values(by='total_subsidies', ascending = False)
    gb['avg_yearly_subsidies'] = (gb.total_subsidies / nb_years).astype(int)
    gb['avg_yearly_nb_requests'] = (gb.nb_requests / nb_years).astype(int)

    # Add traces
    fig.add_trace(
        go.Bar(x=gb.index, y=gb.avg_yearly_subsidies, name='Yearly subsidies (average)'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=gb.index, y=gb.avg_yearly_nb_requests, name='Number of requests'),
        secondary_y=True,
    )

    # Layout
    fig.update_layout(
        title_text='<b>Yearly subsidies and Number of requests by Geography (1-10k range)</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Geography</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Yearly</b> subsidies', color = '#636EFA', secondary_y=False)
    fig.update_yaxes(title_text='<b>Number</b> of requests', color='#EF553B', showgrid = False, tickformat = '.0f', secondary_y=True)

    return fig

fig = geo_paris_idf_beyond_1_10k()

# ...under the hood...
with st.expander('Under the hood...'):
    st.plotly_chart(fig)
    st.caption('''
    Even if we limit the analysis to the 1-10k EUR subsidy range, Paris City Council essentially grants money to Paris-based associations.
    ''')

st.write('Where are located the Parisian-based associations? Let\'s look at those that got subsidies between 1 and 10k EUR.')

@st.cache
def map_1_10k():
    # Dataframes
    data_subset = data.loc[data.montant_vote_scale == 3]
    top5_directions_subset = data_subset.groupby('direction').count().sort_values(by='numero_dossier', ascending = False).index.to_list()[0:5]
    data_subset_top5_directions = data_subset.loc[data_subset.direction.isin(top5_directions_subset)]
    df_map = data_subset_top5_directions.groupby(['siret', 'denomination_unite_legale', 'adresse_etablissement_complete', 'direction']).agg(lat = ('latitude','mean'), lon = ('longitude', 'mean'), total_subsidies = ('montant_vote', 'sum'))
    df_map = df_map.reset_index(level=[1,2,3])


    # Map

    px.set_mapbox_access_token('pk.eyJ1IjoiZS10aW5lcmFudCIsImEiOiJjbDI5MmluZDIwZGU0M2NtZWZ3MGQ2NDdpIn0.HcxDw2oUG2RXOFRZrFyfLQ')
    hover_data={'direction':True,
                'total_subsidies':True,
                'lat':False,
                'lon':False,
                'adresse_etablissement_complete':True
                }
    pt = plotly.colors.qualitative.Plotly
    d3 = plotly.colors.qualitative.D3
    colors = [d3[1], pt[0], d3[7], d3[5], pt[3]]
    fig = px.scatter_mapbox(df_map, lat='lat', lon='lon', hover_name='denomination_unite_legale', hover_data=hover_data,zoom=3, color = 'direction', color_discrete_sequence=colors, height=500, size = 'total_subsidies')
    fig.update_layout(mapbox=dict(bearing=0,center=go.layout.mapbox.Center(lat=48.8523647,lon=2.3482718),pitch=0,zoom=11))
    fig.update_layout(mapbox_style = 'basic')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

fig = map_1_10k()
st.plotly_chart(fig)
st.caption('''
The size of the bubbles denotes the total amount of subsidies received (for subsidies in the 1-10k EUR range).

We can observe that the DDCT, the DPVI and the DASES mostly support associations based in the 10th, 18th, 19th, and 20th arrondissements
while the DJS is more present in the parts of the city built in the 1970s-1980s (as most sport equipments are located there).
The 7th and the 16th arrondissement look quite empty.
''')


@st.cache
def map_reject():
    # Dataframes
    top5_directions = data.groupby('direction').count().sort_values(by='numero_dossier', ascending = False).index.to_list()[0:5]
    df_map = data.loc[data.direction.isin(top5_directions)]
    df_map = df_map.groupby(['siret', 'denomination_unite_legale', 'adresse_etablissement_complete', 'direction'])
    df_map = df_map.agg(lat = ('latitude','mean'), lon = ('longitude', 'mean'), nb_requests = ('numero_dossier', 'count'), nb_requests_success = ('subsidy_granted_bool', 'sum'))
    df_map = df_map.reset_index(level=[1,2,3])
    df_map['reject_rate'] = 1 - df_map.nb_requests_success / df_map.nb_requests
    df_map = df_map.loc[df_map.reject_rate > 0.5]

    # Map

    px.set_mapbox_access_token('pk.eyJ1IjoiZS10aW5lcmFudCIsImEiOiJjbDI5MmluZDIwZGU0M2NtZWZ3MGQ2NDdpIn0.HcxDw2oUG2RXOFRZrFyfLQ')
    hover_data={'direction':True,
                'nb_requests':True,
                'nb_requests_success':False,
                'reject_rate':True,
                'lat':False,
                'lon':False,
                'adresse_etablissement_complete':True
                }
    pt = plotly.colors.qualitative.Plotly
    d3 = plotly.colors.qualitative.D3
    colors = [d3[1], pt[0], d3[7], d3[5], pt[3]]
    fig = px.scatter_mapbox(df_map, lat='lat', lon='lon', hover_name='denomination_unite_legale', hover_data=hover_data,zoom=3, color = 'direction', color_discrete_sequence=colors, height=500, size = 'nb_requests')
    fig.update_layout(mapbox=dict(bearing=0,center=go.layout.mapbox.Center(lat=48.8523647,lon=2.3482718),pitch=0,zoom=11))
    fig.update_layout(mapbox_style = 'basic')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

fig = map_reject()

# ...under the hood...
with st.expander('Under the hood...'):
    st.write('And where are located those that get rejected more than 50% of the time?')

    st.plotly_chart(fig)
    st.caption('''
    The size of the bubbles denotes the total amount of requests.

    The 7th and the 16th arrondissement are not discriminated against: there is just no association requesting subsidies there!
    ''')

st.markdown('---')

# MACHINE LEARNING
st.header('Machine learning: trying to predict the success of a request!', anchor = 'machine_learning')

st.write('Let\'s do a bit of machine learning to assess whether we can predict if a subsidy request is likely to get accepted or not.')

# Data pre-processing & Pipeline
st.subheader('Pipeline', anchor = 'ml-pipeline')
st.write('They say that an image is worth 1,000 words.')

st.image('../06_ML/ML_Pipeline.png')

st.markdown('Wait... ***NLP***? Indeed! In our dataset, we have the description of each request. We used NLP (Spacy librairy) to score the description - the higher the (probabilistic) score, the higher the chance to get a subsidy, *ceteris paribus*...')

# ...under the hood...
with st.expander('Under the hood...'):
    st.write('''
    Apart from NLP applied on the request description, the rest (feature selection, enrichment, encoding) was pretty straightforward.

    Features originally in the dataset include Direction, associations' self-declared domainfields, subsidy's nature, geo-coordinates.

    We enriched them with features such as: Did the association already apply for a subsidy previously? Was it a success? How old was the association at the time of application?
    And of course the NLP scoring.

    In terms of encoding, dummy & binary proved to be the best ways to proceed. NLP scoring was the only continuous variable.
    ''')

# First model
st.subheader('First try', anchor='ml-1')
st.write('We naively fitted a Decision Tree Classifier model (as we imagined it is very close to the way human beings will take a decision in this context), and... BAM.')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label = 'Accuracy (on test data)', value = '92.6%')
with col2:
    st.metric(label = 'Recall (on test data)', value = '93.8%')
with col3:
    st.metric(label = 'Precision (on test data)', value = '94.0%')

st.markdown('***WAY too good to be true...*** :scream:')

with st.expander('Under the hood...'):
    st.write('We also tried a Support Vector Classification and a Logistic Regressor: same results!')

# Reflecting on the first model
st.subheader('Reflecting on the first model', anchor='ml-reflecting')
st.markdown('''
    What's wrong here? It is because of the feature 'nature_subvention', which can take 4 values: investissement, projet, fonctionnement, non précisée.
    Looking at feature importance, these fields are very dominant in the model.
    
    We originally assumed that this was something declared by the association (the documentation did not say a word about it). Actually it appears that this is a field filled by Paris City's services.
    
    And if a subsidy is rejected, they do not bother with this field (it is probably linked with public accounting rules).
''')

@st.cache
def nb_requests_reject_rate_by_nature():
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dataframes
    gb = data.groupby('nature_subvention').agg(nb_dossiers = ('numero_dossier', 'count'))
    ct = pd.crosstab(data.nature_subvention, data.subsidy_granted_bool, normalize='index').reindex(gb.index)

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
        title_text='<b>Number of requests & Reject rate by Nature</b>',
        showlegend = False,
        hovermode='x unified',
    )

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Nature</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>Number</b> of requests', color = '#636EFA', secondary_y=False)
    fig.update_yaxes(title_text='<b>Reject</b> rate', color='#EF553B', showgrid = False, tickformat = '.1%', secondary_y=True)

    return fig

# ...under the hood...
with st.expander('Under the hood...'):
    fig = nb_requests_reject_rate_by_nature()
    st.plotly_chart(fig)
    st.write('''
    Now it is pretty obvious!
    ''')

# Second model
st.subheader('Second try', anchor='ml-2')
st.write('''
    We took out the nature_subvention-related fields.
    We also took out other features that eventually appeared to be not important, such as the direction.
    We ended with only four groups fo features: NLP scoring, existence & success of previous request, localisation in Paris or not, and associations's self-declared fields of activities.

    After bit of :wrench: hyperparameter tuning (GridSearch )...
''')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label = 'Accuracy (on test data)', value = '67.4%')
with col2:
    st.metric(label = 'Recall (on test data)', value = '84.4%')
with col3:
    st.metric(label = 'Precision (on test data)', value = '69.0%')

st.markdown('Not *TOO* bad... and anyway the results did not get any better despite looooong hours :persevere: fine-tuning the whole pipeline.')
st.markdown(':deciduous_tree: Let\'s plot the tree.')

st.image('../06_ML/dtc.png')

st.markdown('''
    Clearly the NLP score is the most important feature. Not a surprise considering that it has been prefit to predict the success of the request!

    But this tree also shows that you are more likely to get a subsidy if the previous one was successful.
    We get a better model by factoring in this feature than just with NLP.
''')
 
with st.expander('Under the hood...'):
    st.write('Again, we also used a Logistic Regressor and a Support Vector Classification models, with basically the same results.')

# Test your request
st.subheader('Sandbox: test your request!', anchor='ml-test_request')
st.write('Please input your request below, and we will tell you if you can hope for a subsidy from Paris City Council!')
st.markdown(':exclamation: :exclamation: :exclamation: *FOR ENTERTAINMENT PURPOSES ONLY* :exclamation: :exclamation: :exclamation:')


with st.form('Your subsidy request'):
    model = st.radio(label = 'Machine Learning Model', options = ['Decision Tree Classifier', 'Logistic Regressor'])
    objet_dossier = st.text_input(label = 'Request description', value = 'un jardin partagé pour les enfants et les personnes âgées')
    secteurs_activites = st.multiselect(label = 'Fields of activities',
                                        options = ['culture_et_arts', 'education_et_formation', 'loisirs', 'precarite_et_exclusion', 'social', 'sport', 'vie_et_animation_locale'],
                                        default = ['culture_et_arts', 'social']
                                        )
    paris = st.checkbox(label = 'Association located in Paris?', value = True)
    previous_request_success = st.radio(label = 'Was your last subsidy request (if any) successful?',
                                        options = ['First request', 'Last request was not successful', 'Last request was successful']
                                        )
    submit_button = st.form_submit_button(label='Submit')

@st.cache
def assess_request(objet_dossier, secteurs_activites, paris, previous_request_success, model):
    
    dct_request_test = {
        'nlp_scoring':'',
        'culture_et_arts':'0',
        'education_et_formation':'0',
        'loisirs':'0',
        'precarite_et_exclusion':'0',
        'social':'0',
        'sport':'0',
        'vie_et_animation_locale':'0',
        'paris':'0',
        'previous_request_success':'0',
        'objet_dossier':''
    }

    dct_request_test['objet_dossier']=objet_dossier
    dct_request_test['paris']= 1 if paris else 0

    if previous_request_success == 'First request':
        dct_request_test['previous_request_success'] = -1
    elif previous_request_success == 'Last request was not successful':
        dct_request_test['previous_request_success'] = 0
    else:
        dct_request_test['previous_request_success'] = 1

    if secteurs_activites:
        for s in secteurs_activites:
            dct_request_test[s] = 1

    if model == 'Decision Tree Classifier':
        model = load('../06_ML/dtc_best.joblib')
    else:
        model = load('../06_ML/lr.joblib')

    nlp_textcat = spacy.load('../05_NLP/textcat_output/model-best')
    dct_request_test['nlp_scoring'] = nlp_textcat(clean_text(dct_request_test['objet_dossier'])).cats['yes']
    df_request = pd.DataFrame.from_dict(dct_request_test, orient = 'index').T
    return model.predict_proba(df_request.drop(['objet_dossier'], axis = 1))[0][1], dct_request_test['nlp_scoring']

assessment, nlp_scoring = assess_request(objet_dossier, secteurs_activites, paris, previous_request_success, model)

try:
    [prev_assessment, prev_nlp_scoring] = joblib.load('prev_scores.joblib')
except:
    prev_assessment, prev_nlp_scoring = assessment, nlp_scoring

dump([assessment, nlp_scoring], 'prev_scores.joblib')

with st.expander('Discover your result!'):
    if assessment > 0.5:
        st.markdown('Your request is likely to be accepted! :thumbsup:')
    else:
        st.markdown('Well, maybe not this time... :cry:')

with st.expander('The magic numbers'):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label = 'Model score', value = f'{assessment:.0%}', delta = f'{assessment - prev_assessment: .0%}')
        st.caption('Overall model probability to see your request accepted.')
    with col2:
        st.metric(label = 'NLP score', value = f'{nlp_scoring:.0%}', delta = f'{nlp_scoring - prev_nlp_scoring: .0%}')
        st.caption('Sweet spot is above 61%, and above 54% if your previous request was successful.')
    st.write('Note that being in Paris or not, and the fields of activities, have absolutely impact with the Decision Tree Classifier. It is different with the Logistic Regressor.')

st.markdown('---')

# MACHINE LEARNING AMOUNT
st.header('Let\'s be more ambitious! Can we predict the amount?', anchor='ml_amount')
st.markdown('''
    
    To try to achieve this, we need to perform additional data transformation, reduction, sampling, model selection:
    - We focused only on the 1-10k EUR subsidy granted to Parisian association (56% of successful requests)
    - We did not try to get an exact amount (using regression models), but a range (using classification models): 1-2k EUR, 3-5k EUR, 6-10k EUR
    - We had to factor in imbalance using oversampling as the 6-10k EUR range was much less populated than the other ones
    - Intuitively, associations that look alike should get the same kinds of amounts, so we decided to opt for the KNeighbors Classifier with hyperparameter tuning :wrench:
''')

with st.expander('And the result is...'):
    st.image('../06_ML/knc.png')
    st.caption('Confusion matrix on test data, normalised. Colors by the artist.')
    st.write('Not very usable due to significant confusion :confused: But it will be a good challenge for those willing to take it!')


st.markdown('---')

# FUN FACTS
st.header('Fun facts', anchor = 'fun_facts')
st.write('A collection of not-so-random facts from our dataset :smiley_cat:')