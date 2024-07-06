# ref原本的逻辑
def category_func_ref_v0(row):
    question = row["问题"]
    response = row["回答"]

    if "招股说明书" in response and "股票数据库" not in response:
        category = "Text"
        raw_category = "Text"
    elif "招股说明书" not in response and "股票数据库" in response:
        category = "SQL"
        raw_category = "SQL"
        for company_name in company_list:
            if company_name in question:
                category = "Text"
    else:
        category = "SQL"
        raw_category = "Unknown"
        for company_name in company_list:
            if company_name in question:
                category = "Text"

    return pd.Series({"分类": category, "原始分类": raw_category})


# 把ref的逻辑重构一下，不改变原意
def category_func_ref_v1(row):
    question = row["问题"]
    response = row["回答"]

    for company_name in company_list:
        if company_name in question:
            category = "Text"
            break
    else:
        if "招股说明书" in response and "股票数据库" not in response:
            category = "Text"
        else:
            category = "SQL"

    if "招股说明书" in response and "股票数据库" not in response:
        raw_category = "Text"
    elif "招股说明书" not in response and "股票数据库" in response:
        raw_category = "SQL"
    else:
        raw_category = "Unknown"

    return pd.Series({"分类": category, "原始分类": raw_category})


# 加规则分类    fxxk，正确率百分百，用啥LLM
def category_func_ref_v2(row):
    question = row["问题"]
    response = row["回答"]

    for company_name in company_list:
        if company_name in question:
            category = "Text"
            break
    else:
        if "招股说明书" in response and "股票数据库" not in response:
            category = "Text"
        else:
            category = "SQL"

    # 纯靠LLM
    if "招股说明书" in response and "股票数据库" not in response:
        raw_category = "Text"
    elif "招股说明书" not in response and "股票数据库" in response:
        raw_category = "SQL"
    else:
        raw_category = "Unknown"

    # 完全靠规则，不依赖LLM
    for company_name in company_list:
        if company_name in question:
            rule_category = "Text"
            break
    else:
        rule_category = "SQL"

    return pd.Series({"分类": category, "原始分类": raw_category, "规则分类": rule_category})
