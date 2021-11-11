def financial_performance_evaluation(prices, labels):    
    """
    Prices: dataframe with true prices
    Labels: labels predicted by model
    """
    available_capital_lst =  [10000]
    available_capital = 10000
    transaction_fee = 5
    fee_sum = 0
    execution_price = 0
    investment_sum = 0
    nr_shares_purchased = 0
    nr_shares_sold = 0
    nr_shares_held = [0]
    transaction_list = [0]

    
    for i, label in enumerate(labels):
        if label == 0 or label == transaction_list[-1]:
            pass

        elif label == 1 and available_capital_lst[-1] > 0:
            execution_price = prices.close.iloc[i]
            investment_sum = available_capital_lst[-1] - transaction_fee
            available_capital_lst.append(available_capital_lst[-1] - investment_sum)
            nr_shares_purchased = investment_sum / execution_price
            nr_shares_held.append(nr_shares_held[-1] + nr_shares_purchased)
            transaction_list.append(label)
            print(f"Day {i}: purchase of {round(nr_shares_purchased,2)} units for total of {round(investment_sum,2)} euros")
            fee_sum += transaction_fee

        elif label == -1 and nr_shares_held[-1] > 0:
            execution_price = prices.close.iloc[i]
            nr_shares_sold = nr_shares_held[-1]
            nr_shares_held.append(nr_shares_held[-1] - nr_shares_sold)
            sale_sum = nr_shares_sold * execution_price - transaction_fee
            available_capital_lst.append(sale_sum)
            transaction_list.append(label)
            print(f"Day {i}: sale of {round(nr_shares_sold,2)} units for total of {round(sale_sum,2)} euros")
            print(f"Return of transaction: {round((available_capital_lst[-1] / available_capital_lst[-3]-1)*100,2)}%")
            print("")
            fee_sum += transaction_fee

    total_final_capital = available_capital_lst[-1] + nr_shares_held[-1] * prices.close.iloc[-1]
    total_return = total_final_capital / available_capital_lst[0] - 1

    print("")
    print(f"End capital on day {len(prices)}: {round(total_final_capital,2)} euros")
    print(f"Total return: {round(total_return*100,2)}%")
    print(f"Shares held at end of period: {round(nr_shares_held[-1],2)}")
    print(f"Total fee spending: {fee_sum}")
