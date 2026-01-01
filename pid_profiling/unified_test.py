from unified_analysis import unified_strategy, transform_path
from execute_query import *
start_server()
hot_functions = unified_strategy(
                #"WITH 50000000 AS N SELECT arrayJoin(topK(500)(intHash64(number) % 1000000)) FROM system.numbers_mt WHERE number < N FORMAT Null;",
                "SELECT id,sin(val) * tan(val) + cos(id % 360) AS math_crunch,SHA256(random_data || toString(math_crunch)) AS heavy_hash FROM stress_test_5 WHERE toString(id) LIKE '\''%5%'\'' ORDER BY math_crunch DESC LIMIT 100;",
                "report",
                False,
                True,
                ["/home/ubuntu/ClickHouse_debug/contrib", "/home/ubuntu/ClickHouse_debug/base/glibc-compatibility/"],
                prefix_path="/home/ubuntu/ClickHouse_debug",
                custom_prefix="/home/ubuntu/ClickHouse_debug",
            )
print(hot_functions)
stop_server()