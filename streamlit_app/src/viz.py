import networkx as nx
import plotly.express as px
# If you move simple charts like create_time_series here later, add pandas
def build_author_subreddit_network(df):
    """
    Builds a bipartite network graph: Authors <-> Subreddits.
    - Red Nodes = Subreddits (Size = Activity volume)
    - Blue Nodes = Authors
    """
    # 1. Filter Data to avoid "Hairball" (Too many nodes)
    top_authors = df['author'].value_counts().head(50).index
    subset = df[df['author'].isin(top_authors)]

    G = nx.Graph()

    # 2. Add Nodes and Edges
    for _, row in subset.iterrows():
        auth = row['author']
        sub = row['subreddit']
        
        if not G.has_node(auth):
            G.add_node(auth, node_type="Author", color='#4DA6FF', size=10) # Small Blue
        
        if not G.has_node(sub):
            G.add_node(sub, node_type="Subreddit", color='#FF4B4B', size=25) # Large Red
            
        G.add_edge(auth, sub)

    # 3. Dynamic Sizing for Subreddits
    for node in G.nodes():
        if G.nodes[node]['node_type'] == "Subreddit":
            degree = G.degree(node)
            G.nodes[node]['size'] = 20 + (degree * 1.5)

    # 4. Generate Layout
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

    # 5. Build Plotly Traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = dict(
        type='scatter',
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node]['node_type']
        node_text.append(f"<b>{node}</b><br>({node_type})")
        
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = dict(
        type='scatter',
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=1,
            line_color='white'
        )
    )

    # 6. Metrics Calculation
    num_subs = len([n for n, d in G.nodes(data=True) if d['node_type'] == 'Subreddit'])
    num_auths = len([n for n, d in G.nodes(data=True) if d['node_type'] == 'Author'])
    num_conns = G.number_of_edges()

    # 7. Create Figure (FIXED LAYOUT SYNTAX)
    fig = dict(data=[edge_trace, node_trace],
               layout=dict(
                   # --- FIX STARTS HERE ---
                   title=dict(
                       text='Author-Subreddit Network',
                       font=dict(size=16)
                   ),
                   # --- FIX ENDS HERE ---
                   showlegend=False,
                   hovermode='closest',
                   margin=dict(b=20, l=5, r=5, t=40),
                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   height=550,
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)'
               ))
    
    return fig, num_subs, num_auths, num_conns