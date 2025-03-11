import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=test.index, y=y_test, 
                         mode='lines', name='Actual Consumption', 
                         line=dict(color='blue')))

fig.add_trace(go.Scatter(x=test.index, y=rf_preds, 
                         mode='lines', name='Random Forest Predictions', 
                         line=dict(color='red', dash='dash')))

fig.add_trace(go.Scatter(x=test.index, y=lstm_preds.flatten(), 
                         mode='lines', name='LSTM Predictions', 
                         line=dict(color='green', dash='dot')))

fig.update_layout(title="Water Consumption Forecasting",
                  xaxis_title="Date", yaxis_title="Consumption",
                  template="plotly_white")

fig.show()
