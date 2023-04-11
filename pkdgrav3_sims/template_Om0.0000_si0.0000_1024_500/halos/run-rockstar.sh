ROCKSTAR=$WORK2/rockstar/rockstar
echo WORKING_DIRECTORY = $(pwd)
echo ROCKSTAR = ${ROCKSTAR}

if [ -f ./restart.cfg ]
then
  ${ROCKSTAR} -c ./restart.cfg &
  sleep 5
  ${ROCKSTAR} -c auto-rockstar.cfg
else
  ls ../snapshots/snapshot.00[0-9][0-9][0-9] | sed -r 's/^.+\.([0-9]+)$/\1/' > ./snapshot_names.txt
  echo "There are $(wc -l ./snapshot_names.txt) snapshots; they are:"
  cat ./snapshot_names.txt

  ${ROCKSTAR} -c ./parallel.cfg &
  echo Waiting for ./auto-rockstar.cfg
  until [ -f ./auto-rockstar.cfg ]
  do
    sleep 1
  done
  echo ./auto-rockstar.cfg is available now.

  ${ROCKSTAR} -c ./auto-rockstar.cfg
fi

