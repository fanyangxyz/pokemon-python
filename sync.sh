while true; do
    rsync -avz --progress ubuntu@146.235.205.115:/home/ubuntu/pokemon-python/batch-results .
    echo "Synced at $(date)"
    sleep 10
done
