import plotly.express as px

def plot_data(data, feature, split_by, plot_type):
    title = f"{plot_type} of {feature} split by {split_by}"
    
    if plot_type == "distribution plot":
        if split_by == "nothing":
            return px.histogram(data, x=feature, title=title)
        else:
            return px.histogram(data, x=feature, color=split_by, title=title)
    
    elif plot_type == "violin plot":
        if split_by == "nothing":
            return px.violin(data, y=feature, box=True, title=title)
        else:
            return px.violin(data, y=feature, box=True, color=split_by, title=title)
        
    else:
        raise ValueError("Unknown plot type")