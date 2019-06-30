def get_parallel_db(db,remove_index):
  return torch.cat((db[0:remove_index],db[remove_index+1:]))
  
def get_parallel_dbs(db):
  parallel_db = list()
  
  for i in range(len(db)):
    pds = get_parallel_db(db,i)
    parallel_db.append(pds)
    
  return parallel_db

def create_db_and_parallels(num_entries):
  db = torch.rand(num_entries) > 0.5
  pdbs = get_parallel_dbs(db)
  
  return db,pdbs

def query(db):
  return db.float().mean()

def sensitivity(query,num_entries=1000):
  db,pdbs = create_db_and_parallels(num_entries)
  sensitive = 0
  
  full_db_result = query(db)
  
  for pdb in pdbs:
    db_distance = torch.abs(query(pdb) - full_db_result)
    
  if(db_distance > sensitive):
    sensitive = db_distance
    
  return sensitive
