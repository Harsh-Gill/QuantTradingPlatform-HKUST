from flask import Flask, render_template
import time

app = Flask(__name__)

def read_last_lines(file_path, n=10):
    with open(file_path) as f:
        lines = f.readlines()
    return lines[-n:]

def read_pnl_data(file_path):
    with open(file_path) as f:
        lines = [line for line in f if "Total PnL" in line]

    return lines[-3:]

@app.route('/')
def index():
    execution_status = read_last_lines('execution_and_status.txt', 10)
    market_data = read_last_lines('market_data_output.txt', 10)
    execution_log = read_last_lines('execution_log.txt', 10)
    submitted_orders = read_last_lines('submitted_orders.txt', 10)
    position_pnl = read_pnl_data('position_and_pnl_log.txt')

    return render_template('index.html', 
                           execution_status=execution_status,
                           market_data=market_data,
                           execution_log=execution_log,
                           submitted_orders=submitted_orders,
                           position_pnl=position_pnl)

if __name__ == '__main__':
    app.run(debug=True)