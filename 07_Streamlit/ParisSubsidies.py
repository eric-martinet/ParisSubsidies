# STREAMLIT APP
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
***Who*** is distributing Paris City Council's subsidies? ***What*** causes is Paris City Council likely to support?
***When*** are these subsidies granted? ***Where*** do they go (i.e. to what associations)? ***Why*** does a non-profit organisation receive a subsididy while another one does not?

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



# ...under the hood...
with st.expander('Under the hood'):
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
st.subheader('Distribution analysis')
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
with st.expander('Under the hood'):
    st.markdown('''
    Given the range of distributed amounts, we used a log10 scale to group the subsidies by size.
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
st.subheader('Geo analysis')
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
with st.expander('Under the hood'):
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
with st.expander('Under the hood'):
    st.write('And where are located those that get rejected more than 50% of the time?')

    st.plotly_chart(fig)
    st.caption('''
    The size of the bubbles denotes the total amount of requests.

    The 7th and the 16th arrondissement are not discriminated against: there is just no association requesting subsidies there!
    ''')

st.markdown('---')

# MACHINE LEARNING
st.header('Machine learning', anchor = 'machine_learning')

st.write('Let\'s do a bit of machine learning to assess whether we can predict if a subsidy request is likely to get accepted or not.')

# First model
st.subheader('First model: trying to predict success of request')