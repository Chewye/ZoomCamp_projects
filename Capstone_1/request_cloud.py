import requests


url = 'http://79.133.181.183:9696/'

data = {'type': 'Участок',
 'weight': 87.0,
 'mailctg': 1,
 'directctg': 2,
 'price_mfi': 150.0,
 'dist_qty_oper_login_1': 42.0,
 'total_qty_oper_login_1': 720176.0,
 'total_qty_oper_login_0': 58950.0,
 'total_qty_over_index_and_type': 779126.0,
 'total_qty_over_index': 8290896.0,
 'is_wrong_rcpn_name': 0,
 'is_wrong_phone_number': 0,
 'oper_type': 1043}


response = requests.post(url, json=data).json()


print(response)