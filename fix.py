python -c "
f=open('main.py',encoding='utf-8')
content=f.read()
f.close()
old='''    if os.path.exists(chroma_meta_path):
        # Collection already built — load from disk
        print(\"\\n[main] ChromaDB collection found on disk. Loading...\")
        chroma_store.load()
    else:
        # First run — embed (reuse if already computed) and build
        print(\"\\n[main] No ChromaDB collection found. Building from scratch...\")

        # Reuse embeddings if already computed for FAISS
        # If FAISS was loaded from disk, embeddings weren\'t computed yet
        if not os.path.exists(FAISS_INDEX_PATH.replace(\".faiss\", \"_docs.txt\")):
            embeddings = embedder.embed_documents(documents)

        # embeddings may already exist from FAISS build above
        try:
            embeddings
        except NameError:
            embeddings = embedder.embed_documents(documents)'''
new='''    try:
        chroma_store.load()
        print(\"\\n[main] ChromaDB collection loaded from disk.\")
    except RuntimeError:
        print(\"\\n[main] No ChromaDB collection found. Building from scratch...\")
        try:
            embeddings
        except NameError:
            embeddings = embedder.embed_documents(documents)
        chroma_store.build(documents, embeddings)'''
content=content.replace(old,new)
f=open('main.py','w',encoding='utf-8')
f.write(content)
f.close()
print('done')
"