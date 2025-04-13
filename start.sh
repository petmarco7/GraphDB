#Bash file to execute before running python code

# Stop any previous neo4j instances
neo4j stop

# Set the project password for the neo4j database
neo4j-admin dbms set-initial-password password

# Start the neo4j database
neo4j start