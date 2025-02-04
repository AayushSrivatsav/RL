#x solvable equation instead of a constant number
#Implementing the algorithm
def algoRankingEqn(Documents, x, users, e=0.1):
  #Assume that the list is already sorted in static order and top few documents have been choosen and sent...
  Rank=[]
  docs = len(Documents)
  for i in range(docs):
    ranks = np.zeros((len(Documents)))
    #Need to finish it
    for j in range(users):
      val = x(e,len(Documents))   #Assume that x is a lambda function which has two inputs epsilon and k - which changes every iteration
      for p in range(round(val)):        #All documents must be presented x times
        doc_index_choosen = np.random.randint(0,len(ranks))     #Choosing by index - hence won't affect
        ranks[doc_index_choosen] = ranks[doc_index_choosen] + 1
    idx = np.argmax(ranks)
    Rank.append(Documents[idx])
    Documents.pop(idx)
  return Rank