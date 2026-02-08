# Answer

**Query:** How can I persist data in Docker containers?

---

To persist data in Docker containers, you can use Docker volumes. A Docker volume allows you to store data outside the container's writable layer, ensuring that your data survives container restarts and removals. Here’s how you can attach a volume to your container:

1. Use the `-v` option with your `docker run` command to specify a volume name and the path where the data should be stored inside the container. If the volume doesn't exist, Docker will automatically create it for you.

   Example command:
   ```console
   $ docker run --name my-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=mydb -v my-db-volume:/var/lib/mysql -d mysql:latest
   ```

2. To verify that the data persists, you can stop and remove the container, then start a new container using the same volume. The data should still be available.

Additionally, you can inspect the volume to see where Docker is storing your data using the `docker volume inspect` command.

Example command:
```console
$ docker volume inspect todo-db
```

This will show you details about the volume, including the mount point on the host system.
