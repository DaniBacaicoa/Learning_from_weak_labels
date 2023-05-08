### This is simply for plotting losses on the simplex
import plotly.figure_factory as ff
import numpy as np

X, Y = np.mgrid[0:1:100j, 0:1:100j]
X, Y = X.ravel(), Y.ravel()
mask = X + Y <= 1
X, Y = X[mask], Y[mask]
Z = 1 - X - Y
Z=Z*(Z>0)
eta = np.array([0.1,0.2,0.7])

loss  = (X-eta[0])**2+(Y-eta[1])**2+(Z-eta[2])**2

#fig = ff.create_ternary_contour(np.array([X,Y,Z]), loss)
#fig.show()
fig = ff.create_ternary_contour(np.array([X,Y,Z]), loss,
                                pole_labels=['(1,0,0)', '(0,1,0)', '(0,0,1)'],
                                ncontours=20,
                                #colorscale='YlGnBu',
                                colorscale='Viridis',
                                #showscale=True,
                                title='Square loss',
                                interp_mode='cartesian')
fig.add_trace(
    go.Scatterternary(
        a=[eta[0],0.4],
        b=[eta[1],0.3],
        c=[eta[2],0.3],
        mode='markers+text',
        text=[r'$\eta$', r'$\hat{\eta}$'],
        textposition='top center',
        marker=dict(size=10, color='red'),
        textfont=dict(size=15, color='white')
    )
)

fig.show()
