
import networkx as nx
import re
import logging
from collections import Counter


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# Prevent messages from being propagated to the root logger (avoids duplicate lines)
logger.propagate = False


__all__ = [
    'SchemaGraphBuilder',
    'build_dalk',
    'build_gr',
    'build_lgraphrag',
    'build_ggraphrag',
    'build_hipporag',
    'build_kgp',
    'build_lightrag',
    'build_raptor',
    'build_tog',
    'GraphBuilderFactory',
]


class SchemaGraphBuilder:
    def build(self, chunks):
        logger.info("=" * 60)
        logger.info("Starting SCHEMA GRAPH BUILD phase")
        logger.info(f"Processing {len(chunks)} chunks")
        
        G = nx.DiGraph()
        logger.debug(f"Created empty DiGraph")
        
        # Add nodes
        logger.info("Phase 1: Adding nodes from chunks...")
        for c in chunks:
            G.add_node(c.chunk_id, kind=c.kind, table=c.table, column=c.column, text=c.text, pk=c.pk)
        logger.info(f"✓ Added {len(G.nodes())} nodes to graph")
        
        # Add table->col edges
        logger.info("Phase 2: Creating table → column schema edges...")
        edge_count = 0
        for u, d in G.nodes(data=True):
            if d['kind'] == 'column':
                t = d['table']
                for tu, td in G.nodes(data=True):
                    if td.get('table') == t and td['kind'] == 'table':
                        G.add_edge(tu, u)
                        edge_count += 1
        logger.info(f"✓ Added {edge_count} table→column edges")
        logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        logger.info("=" * 60)
        
        return G


# --- Helpers ---
def _tokens(s):
    if not s:
        return set()
    return set(re.findall(r"\w+", str(s).lower()))


def _overlap_ratio(a, b):
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(min(len(ta), len(tb)))


def _add_basic_nodes(G, chunks):
    for c in chunks:
        G.add_node(c.chunk_id, kind=c.kind, table=getattr(c, 'table', None), column=getattr(c, 'column', None), text=getattr(c, 'text', ''), pk=getattr(c, 'pk', False))


# --- Method implementations (heuristic/stub implementations) ---
def build_dalk(chunks, overlap_threshold=0.2):
    """Dalk: heuristic linking by token-overlap between chunk texts plus schema edges."""
    logger.info("=" * 60)
    logger.info("Starting DALK GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | overlap_threshold={overlap_threshold}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # schema edges (table -> column)
    logger.info("Phase 2: Creating table → column schema edges...")
    schema_edges = 0
    for u, d in G.nodes(data=True):
        if d.get('kind') == 'column':
            t = d.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges")
    
    # semantic edges by overlap
    logger.info("Phase 3: Creating semantic edges by token-overlap...")
    nodes = list(G.nodes(data=True))
    semantic_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score >= overlap_threshold:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                semantic_edges += 2
    logger.info(f"✓ Added {semantic_edges} semantic edges (threshold={overlap_threshold})")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_gr(chunks, overlap_threshold=0.15):
    """GR: plain Graph-based retrieval — schema edges + light semantic connectivity."""
    logger.info("=" * 60)
    logger.info("Starting GR (Graph Retrieval) GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | overlap_threshold={overlap_threshold}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # schema edges
    logger.info("Phase 2: Creating table → column schema edges...")
    schema_edges = 0
    for u, d in G.nodes(data=True):
        if d.get('kind') == 'column':
            t = d.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges")
    
    # add directed semantic edges
    logger.info("Phase 3: Creating directed semantic edges...")
    semantic_edges = 0
    for u, du in G.nodes(data=True):
        for v, dv in G.nodes(data=True):
            if u == v:
                continue
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score >= overlap_threshold:
                G.add_edge(u, v, weight=score)
                semantic_edges += 1
    logger.info(f"✓ Added {semantic_edges} semantic edges (threshold={overlap_threshold})")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_lgraphrag(chunks, k=3):
    """LGraphRAG (Local search): keep top-k neighbors per node by overlap plus schema."""
    logger.info("=" * 60)
    logger.info("Starting LGraphRAG (Local Search) GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | k={k}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # schema edges
    logger.info("Phase 2: Creating table → column schema edges...")
    schema_edges = 0
    for u, d in G.nodes(data=True):
        if d.get('kind') == 'column':
            t = d.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges")
    
    # compute top-k neighbors by overlap
    logger.info(f"Phase 3: Creating top-{k} semantic neighbors per node...")
    nodes = list(G.nodes(data=True))
    local_edges = 0
    for u, du in nodes:
        scores = []
        for v, dv in nodes:
            if u == v:
                continue
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score > 0:
                scores.append((score, v))
        scores.sort(reverse=True)
        for score, v in scores[:k]:
            G.add_edge(u, v, weight=score)
            local_edges += 1
    logger.info(f"✓ Added {local_edges} local semantic edges (top-k per node)")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_ggraphrag(chunks, overlap_threshold=0.1):
    """GGraphRAG (Global search): connect any reasonably similar nodes globally."""
    logger.info("=" * 60)
    logger.info("Starting GGraphRAG (Global Search) GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | overlap_threshold={overlap_threshold}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    logger.info("Phase 2: Creating global semantic edges...")
    nodes = list(G.nodes(data=True))
    semantic_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score >= overlap_threshold:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                semantic_edges += 2
    logger.info(f"✓ Added {semantic_edges} global semantic edges (threshold={overlap_threshold})")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_hipporag(chunks):
    """HippoRAG: heuristic that emphasizes primary key relationships and intra-table links."""
    logger.info("=" * 60)
    logger.info("Starting HippoRAG GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # table -> column and pk-focused edges
    logger.info("Phase 2: Creating table → column and PK-focused edges...")
    schema_edges = 0
    pk_edges = 0
    for u, du in G.nodes(data=True):
        if du.get('kind') == 'column':
            t = du.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
            if du.get('pk'):
                # link PK columns to other columns in same table
                for v, dv in G.nodes(data=True):
                    if v == u:
                        continue
                    if dv.get('table') == du.get('table') and dv.get('kind') == 'column':
                        G.add_edge(u, v, weight=1.0)
                        pk_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges + {pk_edges} PK-centric edges")
    
    # add light semantic edges
    logger.info("Phase 3: Creating light semantic edges...")
    nodes = list(G.nodes(data=True))
    semantic_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score >= 0.15:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                semantic_edges += 2
    logger.info(f"✓ Added {semantic_edges} light semantic edges")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_kgp(chunks):
    """KGP: knowledge-graph-propagation style — co-occurrence and table-mention edges."""
    logger.info("=" * 60)
    logger.info("Starting KGP (Knowledge Graph Propagation) GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # table->column
    logger.info("Phase 2: Creating table → column schema edges...")
    schema_edges = 0
    for u, d in G.nodes(data=True):
        if d.get('kind') == 'column':
            t = d.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges")
    
    # co-occurrence by table-name mentions in text
    logger.info("Phase 3: Creating co-occurrence edges by table-name mentions...")
    table_names = {n: d.get('table') for n, d in G.nodes(data=True) if d.get('kind') == 'table'}
    name_to_nodes = {}
    for n, tbl in table_names.items():
        if not tbl:
            continue
        name_to_nodes.setdefault(str(tbl).lower(), []).append(n)
    
    cooccurrence_edges = 0
    for u, du in G.nodes(data=True):
        txt = du.get('text', '') or ''
        tokens = _tokens(txt)
        for tblname, nodes in name_to_nodes.items():
            if tblname in tokens:
                for target in nodes:
                    if u != target:
                        G.add_edge(u, target, weight=0.9)
                        cooccurrence_edges += 1
    logger.info(f"✓ Added {cooccurrence_edges} co-occurrence edges")
    
    # fallback semantic edges by overlap
    logger.info("Phase 4: Creating fallback semantic edges...")
    nodes = list(G.nodes(data=True))
    semantic_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score > 0.0:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                semantic_edges += 2
    logger.info(f"✓ Added {semantic_edges} semantic edges")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_lightrag(chunks, strong_threshold=0.5):
    """LightRAG: very lightweight — only schema and strong semantic matches."""
    logger.info("=" * 60)
    logger.info("Starting LightRAG GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | strong_threshold={strong_threshold}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    # schema
    logger.info("Phase 2: Creating table → column schema edges...")
    schema_edges = 0
    for u, d in G.nodes(data=True):
        if d.get('kind') == 'column':
            t = d.get('table')
            for tu, td in G.nodes(data=True):
                if td.get('table') == t and td.get('kind') == 'table':
                    G.add_edge(tu, u)
                    schema_edges += 1
    logger.info(f"✓ Added {schema_edges} schema edges")
    
    # strong semantic
    logger.info(f"Phase 3: Creating strong semantic edges (threshold={strong_threshold})...")
    nodes = list(G.nodes(data=True))
    strong_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score >= strong_threshold:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                strong_edges += 2
    logger.info(f"✓ Added {strong_edges} strong semantic edges")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_raptor(chunks, prune_quantile=0.5):
    """RAPTOR: build weighted semantic graph, run PageRank and prune weaker edges."""
    logger.info("=" * 60)
    logger.info("Starting RAPTOR GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks | prune_quantile={prune_quantile}")
    
    G = nx.DiGraph()
    _add_basic_nodes(G, chunks)
    logger.info(f"✓ Phase 1: Added {len(G.nodes())} nodes")
    
    logger.info("Phase 2: Building weighted semantic graph...")
    nodes = list(G.nodes(data=True))
    weights = []
    semantic_edges = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score > 0:
                G.add_edge(u, v, weight=score)
                G.add_edge(v, u, weight=score)
                weights.append(score)
                semantic_edges += 2
    logger.info(f"✓ Added {semantic_edges} initial semantic edges")
    
    if len(G) == 0:
        logger.info("✓ Graph is empty, returning")
        return G
    
    logger.info("Phase 3: Running PageRank algorithm...")
    pr = nx.pagerank(G, weight='weight')
    logger.info(f"✓ PageRank computed for {len(pr)} nodes")
    
    logger.info(f"Phase 4: Pruning edges by quantile threshold ({prune_quantile})...")
    pr_values = sorted(pr.values())
    if not pr_values:
        logger.info("✓ No PR values, returning full graph")
        return G
    
    cutoff_index = int(len(pr_values) * prune_quantile)
    cutoff = pr_values[cutoff_index]
    
    to_remove = []
    for u, v, d in G.edges(data=True):
        if (pr.get(u, 0.0) + pr.get(v, 0.0)) / 2.0 < cutoff:
            to_remove.append((u, v))
    
    for u, v in to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    logger.info(f"✓ Pruned {len(to_remove)} edges (cutoff={cutoff:.6f})")
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    logger.info("=" * 60)
    return G


def build_tog(chunks):
    """ToG: build a maximum-spanning-tree over semantic similarities to create a compact backbone."""
    logger.info("=" * 60)
    logger.info("Starting ToG (Tree of Graphs) GRAPH BUILD phase")
    logger.info(f"Processing {len(chunks)} chunks")
    
    # build undirected weighted graph
    logger.info("Phase 1: Building undirected weighted graph from semantic similarities...")
    U = nx.Graph()
    for c in chunks:
        U.add_node(c.chunk_id, kind=c.kind, table=getattr(c, 'table', None), text=getattr(c, 'text', ''))
    logger.info(f"✓ Added {len(U.nodes())} nodes")
    
    logger.info("Phase 2: Computing semantic weights...")
    nodes = list(U.nodes(data=True))
    edges_added = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, du = nodes[i]
            v, dv = nodes[j]
            score = _overlap_ratio(du.get('text', ''), dv.get('text', ''))
            if score > 0:
                U.add_edge(u, v, weight=score)
                edges_added += 1
    logger.info(f"✓ Added {edges_added} weighted edges")
    
    if U.number_of_edges() == 0:
        logger.info("⚠ No edges found, returning empty directed graph")
        return nx.DiGraph()
    
    logger.info("Phase 3: Computing maximum spanning tree...")
    T = nx.maximum_spanning_tree(U, weight='weight')
    logger.info(f"✓ MST has {len(T.nodes())} nodes, {len(T.edges())} edges")
    
    # convert to directed graph (bidirectional edges in tree)
    logger.info("Phase 4: Converting to directed graph with bidirectional edges...")
    G = nx.DiGraph()
    for n, d in T.nodes(data=True):
        G.add_node(n, **d)
    for u, v, d in T.edges(data=True):
        G.add_edge(u, v, weight=d.get('weight'))
        G.add_edge(v, u, weight=d.get('weight'))
    logger.info(f"✓ Final graph: {len(G.nodes())} nodes, {len(G.edges())} edges (bidirectional tree)")
    logger.info("=" * 60)
    return G


# --- Factory for selecting graph builder method ---
class GraphBuilderFactory:
    """Factory to select and build graphs using different methodologies."""
    
    def __init__(self):
        self.methods = {
            'schema': SchemaGraphBuilder().build,
            'dalk': build_dalk,
            'gr': build_gr,
            'lgraphrag': build_lgraphrag,
            'ggraphrag': build_ggraphrag,
            'hipporag': build_hipporag,
            'kgp': build_kgp,
            'lightrag': build_lightrag,
            'raptor': build_raptor,
            'tog': build_tog,
        }
    
    def build_graph(self, method: str, chunks, **kwargs):
        """
        Build a graph using the specified method.
        
        Args:
            method: Name of the method ('schema', 'dalk', 'gr', 'lgraphrag', 'ggraphrag', 
                   'hipporag', 'kgp', 'lightrag', 'raptor', 'tog')
            chunks: List of chunk objects
            **kwargs: Additional arguments to pass to the builder (e.g., overlap_threshold, k, etc.)
        
        Returns:
            NetworkX DiGraph
        
        Raises:
            ValueError: If method is not recognized
        """
        if method not in self.methods:
            logger.error(f"❌ Unknown method: {method}")
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        logger.info(f"🔨 Selected graph builder method: '{method}'")
        builder = self.methods[method]
        return builder(chunks, **kwargs)
    
    def get_available_methods(self):
        """Return list of available builder methods."""
        return list(self.methods.keys())

