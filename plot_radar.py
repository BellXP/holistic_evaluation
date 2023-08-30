import plotly.graph_objects as go

# categories = ['processing cost','mechanical properties','chemical stability',
#               'thermal stability', 'device integration']

# 6x12
# matrix = [
#     [42.8, 41.8, 41.3, 33.3, 28.8, 30.8, 30.8, 34.8, 35.5, 37.0, 28.8, 54.5],
#     [52.9, 57.0, 25.7, 14.4, 11.0, 11.4, 18.1,  6.7, 15.1,  2.7, 44.1, 60.6],
#     [47.5, 51.5, 48.0, 39.5, 28.4, 20.5, 41.3, 28.0, 37.3, 33.3, 34.0, 60.7],
#     [45.2, 65.2, 58.0, 46.8, 35.2, 23.6, 45.6, 45.2,  4.0, 50.4, 49.6, 58.0],
#     [82.7, 88.7, 52.0, 46.7, 58.0, 64.3, 50.7, 44.3, 70.0, 52.0, 75.3, 67.3],
#     [50.4, 58.6, 74.4, 67.4, 61.8, 60.4, 39.4, 58.2, 76.0, 78.8, 60.4, 83.6],
# ]

# 12x6
# data = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

data = [
    [59.5,52.85714286,47.45454545,45.2,82.66666667,50.48],
    [58.25,57,51.45454545,65.2,88.66666667,58.56],
    [51.75,25.71428571,48.18181818,58,52,74.4],
    [40.75,14.42857143,39.63636364,46.8,46.66666667,67.48],
    [41.25,11,28.36363636,35.2,58,61.76],
    [44.5,11.42857143,20.54545455,23.6,64.33333333,60.4],
    [42.5,18.14285714,41.45454545,45.6,50.66666667,39.48],
    [36.5,6.714285714,28.72727273,45.2,44.33333333,58.12],
    [41,15.14285714,40.18181818,52.8,70,76.04],
    [41.25,2.714285714,33.27272727,50.4,52,78.8],
    [43,44.14285714,33.81818182,49.6,75.33333333,60.44],
    [54.5,60.57142857,62.18181818,58,67.33333333,83.6],
]
index = [
    'Visual Perception', 'Visual Knowledge Acquisition', 'Visual Reasoning',
    'Visual Commonsense','Object Hallucination','Embodied Intelligence'
]
columns = [
    'BLIP2', 'InstructBLIP', 'LLaMA-Adapter V2', 'LLaVA', 
    'MiniGPT-4', 'mPLUG-Owl', 'OpenFlamingo V2', 'Otter', 
    'Otter-I', 'PandaGPT', 'VPGTrans', 'Bard'
]
colors = [
    'rgb(0,255,255)', 'rgb(112,128,144)', 'rgb(210,180,140)', 'rgb(128,0,128)',
    'rgb(255,20,147)', 'rgb(173,255,47)', 'rgb(75,0,130)', 'rgb(255,69,0)',
    'rgb(255,215,0)', 'rgb(160,82,45)', 'rgb(106,90,205)', 'rgb(255,0,0)'
]

fig = go.Figure()

for i in range(len(columns)):
    fig.add_trace(go.Scatterpolar(
        r=data[i],
        theta=index,
        fill='toself',
        name=columns[i],
        opacity=0.45,
        marker=dict(color=colors[i], size=2)
    ))
    # if columns[i] == 'Bard':
    #     fig.add_trace(go.Scatterpolar(
    #         r=data[i],
    #         theta=index,
    #         fill='toself',
    #         name=columns[i],
    #         opacity=0.45,
    #         marker=dict(
    #             color=colors[i],
    #             size=2,)
    #     ))
    # else:
    #     fig.add_trace(go.Scatterpolar(
    #         r=data[i],
    #         theta=index,
    #         fill='toself',
    #         name=columns[i],
    #         opacity=0.45,
    #         marker=dict(
    #             color=colors[i],
    #             size=2,)
    #     ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 90],
      tick0 = 0, 
      dtick = 20,
    ),
    angularaxis = dict(tickfont = dict(size = 14),)
    ),
  showlegend=True,
  legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1,
    xanchor="right",
    x=1
    )
)
fig.write_image("tiny_ehub_radar.pdf")
#fig.show()
