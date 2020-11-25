def histogram_plot(df,fet):
    from plotly.offline import iplot
    import plotly.graph_objs as go
    import plotly.express as px
    trace = go.Histogram(
                    x=df[fet],
                    name=fet,  # name used in legend and hover labels
                    marker=dict(color='#0e918c')
                    )
    data = [trace]
    layout = go.Layout(
                   barmode='overlay',
                   title="HISTOGRAM : {} ".format(fet),
                   xaxis=dict(title=fet),
                   yaxis=dict(title='COUNT')
                   )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor' : 'rgba(0,0,0,0)'})
    fig.update_layout(title_text="<b>HISTOGRAM<b> : {} ".format(fet), title_x=0.5)
    
    fig.update_xaxes(title_text='<b>' + fet +'<b>')
    fig.update_yaxes(title_text='<b>COUNT<b>')
    iplot(fig)
  
#%%
def density(df,fet):
    from plotly.offline import iplot
    import plotly.figure_factory as ff
    from plotly.figure_factory import create_2d_density
    fig = ff.create_2d_density(x=df[fet].value_counts().keys().tolist(), y=df[fet].value_counts().tolist())
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor' : 'rgba(0,0,0,0)'})
    fig.update_layout(title_text="<b>2D Density Plot<b> : {} ".format(fet), title_x=0.5)
    iplot(fig) 
    
#%%
def count_plot(df,fet):
    from plotly.offline import iplot
    import plotly.graph_objs as go
    trace = go.Bar(
                    x=df[fet].value_counts().keys().tolist(),
                    y=df[fet].value_counts().tolist(),
                    marker=dict(color='#0e918c') 
                    )
    data = [trace]
    layout = go.Layout(
                   barmode='overlay',
                   title="COUNT PLOT : {} ".format(fet),
                   xaxis=dict(title=fet),
                   yaxis=dict( title='COUNT')
                   )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor' : 'rgba(0,0,0,0)'})
    fig.update_layout(title_text="<b>COUNT PLOT<b> : {} ".format(fet), title_x=0.5)
    fig.update_xaxes(title_text='<b>' + fet +'<b>')
    fig.update_yaxes(title_text='<b>COUNT<b>')
    iplot(fig)   
    
#%% Basic statistics for continuous column
def basic_stats(df,fets):
    import pandas as pd
    from tabulate import tabulate
    from prettytable import PrettyTable
    from termcolor import cprint
    import statistics
    import scipy.stats
    cprint('SUMMARY STATISTICS : ', attrs=['bold'])
    print()
    #fets=[input("Enter the column name")]
    features=fets
    #features=(['5% quantile','95% quanrtile','skewness','kurtosis','variance','standard deviation'])
    basic_stat=pd.DataFrame(columns=features)
    basic_stats.loc['MAX']= df.max()
    basic_stats.loc['MIN']= df.min()
    basic_stat.loc['Range']=df[fets].max()-df[fets].min()
    basic_stat.loc['skewness']= df[fets].skew()
    basic_stat.loc['kurtosis']= df[fets].kurtosis()
    basic_stat.loc['variance']= df[fets].var()
    basic_stat.loc['standard deviation']= df[fets].std()
    d=basic_stat
    display(d)

def percentile(df,fets):
    import pandas as pd
    from tabulate import tabulate
    from prettytable import PrettyTable
    from termcolor import cprint
    import statistics
    cprint('Percentile Table : ', attrs=['bold'])
    print()
    #fets=[input("Enter the column name")]
    features=fets
    percentile_table=pd.DataFrame(columns=features)
    percentile_table.loc['5% quantile']=df[fets].quantile(0.05)
    percentile_table.loc['10% quantile']=df[fets].quantile(0.1)
    percentile_table.loc['20% quantile']=df[fets].quantile(0.2)
    percentile_table.loc['30% quantile']=df[fets].quantile(0.3)
    percentile_table.loc['40% quantile']=df[fets].quantile(0.4)
    percentile_table.loc['50% quantile']=df[fets].quantile(0.5)
    percentile_table.loc['60% quantile']=df[fets].quantile(0.6)
    percentile_table.loc['70% quantile']=df[fets].quantile(0.7)
    percentile_table.loc['80% quantile']=df[fets].quantile(0.8)
    percentile_table.loc['90% quantile']=df[fets].quantile(0.9)
    percentile_table.loc['95% quantile']=df[fets].quantile(0.95)
    d=percentile_table
    display(d)    

#%%   Basic statistics for categorical column 

def basic_cat_stats(df,fets):
    import pandas as pd
    from tabulate import tabulate
    from prettytable import PrettyTable
    from termcolor import cprint
    import statistics
    import scipy.stats
    cprint('SUMMARY STATISTICS : ', attrs=['bold'])
    print()
    #fets=[input("Enter the column name")]
    features=fets
    #features=(['5% quantile','95% quanrtile','skewness','kurtosis','variance','standard deviation'])
    basic_stat=pd.DataFrame(columns=features)
    basic_stat.loc['DATATYPE']= df.dtypes
    basic_stat.loc['MISSING VALUES']= df.isna().sum()
    basic_stat.loc['NUMBER OF ZEROS']=df.isin([0]).sum()
    basic_stat.loc['COUNT']= df.count()
    basic_stat.loc['UNIQUE']= df.nunique()
    d=basic_stat
    display(d)
    
#%%Distribution plot
def bdist_plot(df,fet):
    import pandas as pd
    import numpy as np
    #df=pd.read_csv('D:\Solytics\Datasets\scores.csv')
    fets=list (df.columns)
    numcol= list ()
    for fet in fets:
        if df[fet].dtype !='O':
            numcol.append(fet)
    print('CHOOSE 1 COLUMN FROM THE LIST:')
    print('THE OPTIONS ARE:')
    print(numcol)
    ch=input('ENTER:')
    import plotly.figure_factory as ff
    hist_data=[df[ch]]
    group_labels = [ch]
    fig = ff.create_distplot(hist_data,group_labels,colors=['red','blue','green']) #Change colours with palette)
                         #title='Distplot',
                         #template='simple_white')
    fig['layout'].update(title='Distribution Plot: {}'.format(ch))
    fig.show()    
    
#%% pareto plot

def pareto_cat(df,fet):
  import pandas as pd
  import numpy as np
  import plotly.graph_objects as go
  fets=list (df.columns)
  catcol=list ()
  for fet in fets:
    if df[fet].dtype =='O' :
      catcol.append(fet)
  print('CHOOSE 1 COLUMN FROM THE LIST:')
  print('THE OPTIONS ARE:')
  print(catcol)
  ch=input('ENTER:')
  p=list(np.unique(df[ch].astype(str)))
  freq=dict()
  a=df[ch].value_counts()
  for i in range(len(p)):
    freq[p[i]]=a[i]
  df1=pd.DataFrame.from_dict(freq,orient='index',columns=['Frequency'])
  cum=list ()
  s=0
  whole=np.sum(df1['Frequency'])
  for i in range(len(p)):
    s=s+df1['Frequency'][p[i]]
    per=(s/whole)*100
    cum.append(per)
  df1['Cumulative Frequency Percentage']=cum
  trace1 = dict(type='bar',
                x=p,
                y=df1['Frequency'],
                marker=dict(
                    color='#2196F3'
                    ),
                name=ch,
                opacity=0.8
                )
  trace2 = dict(type='scatter',
                x=p,
                y=df1['Cumulative Frequency Percentage'],
                marker=dict(
                    color='#263238'
                    ),
                line=dict(
                    color= '#263238', 
                    width= 1.5),
                xaxis='x1', 
                yaxis='y2' 
                )
  data = [trace1, trace2]
  layout = go.Layout(
      title='Pareto Analysis',
      legend= dict(orientation="h"),
      yaxis=dict(
          title='Frequency',
          titlefont=dict(
              color="#2196F3"
              )
          ),
          yaxis2=dict(
              title='Cumulative %',
              titlefont=dict(
                  color='#263238'
                  ),
                  range=[0,105],
                  overlaying='y',
                  anchor='x',
                  side='right'
                  )
          )
  fig = go.Figure(data=data, layout=layout)
  fig.show()  

#%%  
    