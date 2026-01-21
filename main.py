
import argparse
import os
import logging
from Data.Spider.SpiderLoader import SpiderLoader
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Graph.GraphBuilder import GraphBuilderFactory
from Core.Retriever.SchemaRetriever import SchemaRetriever
from Core.LLM.SQLGenerator import SQLGenerator
# from Core.LLM.OpenAIClient import OpenAIClient
from Core.LLM.OllamaLLMClient import OllamaLLMClient as OpenAIClient


# Configure logging for main script
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def run_text2sql(root, split, out_dir, method='schema', max_examples=None, nosql_target='mongodb'):
    logger.info("=" * 80)
    logger.info("📊 TEXT-TO-SQL PIPELINE STARTED")
    logger.info("=" * 80)
    
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"✓ Output directory ready: {out_dir}")
    
    # Phase 0: Initialize
    logger.info("\n[Phase 0] Initializing components...")
    loader = SpiderLoader(root)
    logger.info(f"  ✓ SpiderLoader initialized")
    
    fac = ChunkFactory()
    logger.info(f"  ✓ ChunkFactory initialized")
    
    builder_factory = GraphBuilderFactory()
    available_methods = builder_factory.get_available_methods()
    logger.info(f"  ✓ GraphBuilderFactory initialized | Available methods: {available_methods}")
    
    llm = OpenAIClient()
    logger.info(f"  ✓ LLM client initialized")
    
    gen = SQLGenerator(llm, nosql_target=nosql_target)
    logger.info(f"  ✓ SQLGenerator initialized")
    logger.info("✅ All components initialized successfully\n")
    
    # Validate method
    if method not in available_methods:
        logger.error(f"❌ Unknown method '{method}'")
        logger.error(f"Available methods: {available_methods}")
        return
    logger.info(f"🔨 Using graph builder method: '{method}'\n")
    
    # Phase 1: Load examples
    logger.info("[Phase 1] Loading dataset examples...")
    examples = list(loader.load_examples(split=split))
    logger.info(f"✓ Loaded {len(examples)} examples from split '{split}'\n")
    
    # Process each example
    for idx, ex in enumerate(examples, 1):
        logger.info("=" * 80)
        logger.info(f"Processing example {idx}/{len(examples)} | ID: {ex.get('id', 'unknown')}")
        logger.info("=" * 80)
        
        # Phase 2: Chunking
        logger.info("\n[Phase 2] CHUNKING - Converting database schema to chunks...")
        chunks = fac.schema_to_chunks(ex['db_schema'])
        logger.info(f"✓ Created {len(chunks)} chunks from schema\n")
        
        # Phase 3: Graph Building
        logger.info("[Phase 3] GRAPH BUILDING - Building knowledge graph...")
        try:
            G = builder_factory.build_graph(method, chunks)
            logger.info(f"✓ Graph building complete: {len(G.nodes())} nodes, {len(G.edges())} edges\n")
        except Exception as e:
            logger.error(f"❌ Graph building failed: {e}")
            continue
        
        # Phase 4: Retrieval
        logger.info("[Phase 4] RETRIEVAL - Extracting relevant subgraph...")
        try:
            _, sub = SchemaRetriever(G).retrieve(ex['question'])
            logger.info(f"✓ Retrieved relevant subgraph | Question: {ex['question']}...\n")
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            continue
        
        # Phase 5: SQL Generation
        logger.info("[Phase 5] SQL GENERATION - Generating SQL query...")
        try:
            sql = gen.generate_sql(ex['question'], sub)
            logger.info(f"✓ SQL generated successfully")
            logger.info(f"  SQL: {sql[:100]}...\n")
        except Exception as e:
            logger.error(f"❌ SQL generation failed: {e}")
            continue
        
        # Phase 5b: NoSQL Generation
        logger.info("[Phase 5b] NOSQL GENERATION - Translating SQL to NoSQL...")
        try:
            nosql = gen.generate_nosql(sql, sub, target=nosql_target)
            logger.info(f"✓ NoSQL generated successfully ({nosql_target})")
            logger.info(f"  NoSQL: {nosql[:100]}...\n")
        except Exception as e:
            logger.error(f"❌ NoSQL generation failed: {e}")
            nosql = None
        
        # Phase 6: Save output
        logger.info("[Phase 6] PERSISTENCE - Saving results...")
        sql_file = os.path.join(out_dir, f"{ex.get('id', 'q')}.sql")
        with open(sql_file, 'a') as f:
            f.write(sql + '\n')
        logger.info(f"✓ SQL appended to: {sql_file}")

        if nosql:
            nosql_file = os.path.join(out_dir, f"{ex.get('id', 'q')}.nosql")
            with open(nosql_file, 'a') as f:
                f.write(nosql + '\n')
            logger.info(f"✓ NoSQL appended to: {nosql_file}\n")
        else:
            logger.info("⚠️ NoSQL not generated for this example\n")
        
        logger.info("=" * 80)
        logger.info(f"✅ Example {idx}/{len(examples)} completed")
        logger.info("=" * 80)
        
        if max_examples and idx >= max_examples:
            logger.info(f"🛑 Reached max examples limit ({max_examples}). Stopping...")
            break
    
    logger.info("=" * 80)
    logger.info("✅ TEXT-TO-SQL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Text-to-SQL pipeline with multiple graph methods')
    p.add_argument('--task', required=True, help='Task to run (e.g., text2sql)')
    p.add_argument('--spider_root', required=True, help='Path to Spider dataset')
    p.add_argument('--split', default='dev', help='Dataset split (default: dev)')
    p.add_argument('--method', default='schema', 
                   help='Graph builder method: schema, dalk, gr, lgraphrag, ggraphrag, hipporag, kgp, lightrag, raptor, tog (default: schema)')
    p.add_argument('--nosql_target', default='mongodb',
                   help='Target NoSQL dialect (e.g., mongodb, dynamodb, cosmosdb). Default: mongodb')
    p.add_argument('--max_examples', type=int, help='Maximum number of examples to process (default: all)')
    args = p.parse_args()
    
    if args.task == 'text2sql':
        run_text2sql(args.spider_root, args.split, 'outputs_text2sql', args.method, args.max_examples, args.nosql_target)
    else:
        logger.error(f"Unknown task: {args.task}")
