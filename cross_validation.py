"""
K-fold cross validation
"""

from random import randint
import numpy
 
# Random split dataset into train and test
def get_train_validation_test( dataset, X_column, split=0.8, fold=4):
  """
  shuffle dataset and returns train, test and original 
  """
  print( 'Dataset shape = {}'.format( dataset.shape ) )
  tot_size   = dataset.shape[0]
  print( 'tot size = {}'.format( tot_size ) )
  train_size = int(split * tot_size)
  chunk_size = int( float(train_size)/fold )
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
  print( 'List of chunks has {} chunks of size = {}'.format(len(list_of_chunks), [len(chunk) for chunk in list_of_chunks] ))
  test   = numpy.array(copy_list)
  X_test = test[:, :X_column]
  y_test = test[:, X_column:]
  TEST   = [ X_test, y_test ]
 
  train_dict = dict() 
  for ii in range(fold):
      tmp_list_of_chunks = list( list_of_chunks )
      tmp_valid          = numpy.array(tmp_list_of_chunks[ii])
      X_valid = tmp_valid[:, :X_column]
      y_valid = tmp_valid[:, X_column:]
      VALID   = [ X_valid, y_valid ]
      tmp_list_of_chunks.pop( ii )
      tmp_train          = numpy.concatenate( tmp_list_of_chunks, axis=0 )
      X_train = tmp_train[:, :X_column]
      y_train = tmp_train[:, X_column:]
      TRAIN   = [ X_train, y_train ]
      train_dict[ii] = [ TRAIN, VALID, TEST ]

  return( train_dict )
