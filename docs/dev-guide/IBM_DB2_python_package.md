## IBM DB2 数据库 python 连接

### 1. ibm_db package 下载

```shell
pip install ibm_db
```

install 的时候可能会报错，官方安装指南见

https://github.com/ibmdb/python-ibmdb#installation

运行程序可能报错找不到clidriver/bin，这时候需要安装db的客户端，然后把clidriver文件夹复制到Python的site-packages下面

### 2. 在本地使用 ibm_db 连接数据库

```python
import ibm_db_dbi
import pandas as pd

conn = ibm_db_dbi.connect(database, username, password)

selected_fields = ['ID', 'USER_NAME']
sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)

df=pd.read_sql(sql,conn)
print(df)
conn.close()
```

df 是一个 dataframe，不需要 cursor 的连接

#### 重点是：1. 这里没有默认端口；2. 这里没有主机的参数设置

#### 也即没有 host 和 port 的参数设置

官方示例代码链接如下：

https://www.ibm.com/docs/en/db2/11.5?topic=db-connecting-database-server



Github API 示例代码如下：

```python
import ibm_db
#use connection string
conn=ibm_db.connect("DATABASE=database;HOSTNAME=hostname;PORT=port;PROTOCOL=TCPIP;UID=username;PWD=password",'','')

#use connection string with JWT access token
conn=ibm_db.connect("DATABASE=database;HOSTNAME=hostname;PORT=port;accesstoken=complete_access_token;authentication=token;accesstokentype=jwt;",'','')

#use options
options = { ibm_db.SQL_ATTR_INFO_PROGRAMNAME : 'TestProgram', ibm_db.SQL_ATTR_CURRENT_SCHEMA : 'MYSCHEMA' }
conn=ibm_db.connect("DATABASE=database;HOSTNAME=hostname;PORT=port;PROTOCOL=TCPIP;UID=username;PWD=password",'','', options)

Note: Local cataloged database implicit connection
i) If database parameter specified is a local database alias name with blank userid and password
then connect/pconnect API will use current logged in user's userid for implicit connection
eg: **conn = ibm_db.connect('sample', '', '')**

ii) If database parameter is a connection string with value "DSN=database_name" then
connect/pconnect API will use current logged in user's userid for implicit connection
eg: **conn = ibm_db.connect('DSN=sampledb', '', '')**
If you are using DSN in connection string as in above example, then you must specify other necessary connection details like hostname, userid, password via supported keywords in db2dsdriver.cfg configuration file located under site-packages/clidriver/cfg or under the cfg folder as per the path you have set IBM_DB_HOME to. You can refer to the sample file below.
For more information, please refer [IBM data server driver configuration keywords](https://www.ibm.com/support/knowledgecenter/en/SSEPGG_11.1.0/com.ibm.swg.im.dbclient.config.doc/doc/c0054698.html).
```

尝试后仍未连接成功



因此最后的代码是

```python
def from_db2db(cls, role, dataset_type,
                    host, user, password, database, table,
                    *,
                    target_fields=None, excluding_fields=False,
                    mappings=None, transform=None, port=None
    ):
        """
            Load dataset from IBM DB2 database.
            No default port
        """
        import ibm_db_dbi

        connection = ibm_db_dbi.connect(database, user, password)

        selected_fields = cls._get_selected_fields(
                    db_type='db2',
                    cursor=None,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                    conn=connection
                )
        sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)

        df_dataset = pd.read_sql(sql,connection)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform
        )

```

### 3. 表头的获取

ibm_db 没有 fetchall() 的接口，具体 fetch 函数链接如下：

https://www.ibm.com/docs/en/db2/11.5?topic=db-fetching-rows-columns-from-result-set

我们使用下列文件

```python
import ibm_db

conn = ibm_db.connect(database, username, password)

sql = "SELECT * FROM {}".format(table)
stmt = ibm_db.exec_immediate(conn, sql)

result = ibm_db.fetch_both(stmt)
'''
Returns a dictionary, which is indexed by both column name and position, representing a row in a result set.
result 为取出来的一行的数据
e.g. result = {'ID':1, 0:1, 'USER_NAME':'xxx', 1:'xxx'}
'''

keys = list(result.keys())
all_fields = keys[::2]
 # all_fields = ['ID', 'USER_NAME'] 得到表头
```

因此 _get_selected_fields 函数改动如下：

```python
def _get_selected_fields(db_type, cursor, table, target_fields, excluding_fields, conn=None):
        if db_type == 'db2':
            import ibm_db
            sql = "SELECT * FROM {}".format(table)
            stmt = ibm_db.exec_immediate(conn, sql)
            result = ibm_db.fetch_both(stmt)
            # result = {'ID':1, 0:1, 'USER_NAME':'xxx', 1:'xxx'}
            keys = list(result.keys())
            all_fields = keys[::2]
            # all_fields = ['ID', 'USER_NAME']
        else:
            if db_type == "oracle":
                sql = "select * from {} fetch first 1 rows only".format(table)
            else:
                sql = "select * from {} limit 1".format(table)
            cursor.execute(sql)
            # description is a tuple of tuple,
            # the first position of tuple element is the field name
            all_fields = [tuple_[0] for tuple_ in cursor.description]
```

这里单独把 db2 列出来是因为 db2 获取表头不需要 cursor，但需要 connection 参数（我们在 _get_selected_fields 参数中添加 conn=None）
