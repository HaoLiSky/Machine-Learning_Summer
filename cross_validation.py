"""
K-fold cross validation
"""

from random import randint
import numpy
 
# Random split dataset into train and test
def kfold_cross_validation( dataset, split=0.8, fold=4):
  """
  shuffle dataset and returns train, test and original 
  """
  tot_size   = dataset.shape[0]
  train_size = round(split * tot_size)
  chunk_size = int( float(train_size)/fold )
  print( 'Dataset shape = {}'.format( dataset.shape ) )
  print( 'Training set  = {}'.format( train_size ) )
  
  numpy.random.shuffle( dataset )
  copy_list  = list( dataset )
  list_of_chunks = []
  for ii in range(fold):
    print( 'k fold {}/{}'.format(ii+1,fold) )
    chunk = []
    while len(chunk) < chunk_size:
      index = randint(0, len(copy_list)-1 )
      chunk.append( copy_list[index] )
      copy_list.pop( index )
    print( 'chunk size = {}'.format( len(chunk) ))
    list_of_chunks.append( chunk )
   
  print( 'dataset is now = {}'.format( numpy.array(copy_list).shape ))
  return( list_of_chunks )
