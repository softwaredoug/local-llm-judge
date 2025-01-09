import phi_3_vision_mlx as pv


def describe(query, product):

    agent = pv.Agent(quantize_model=True, quantize_cache=True)

    agent("Step by step, describe the information need behind the query: " + query)

    product_prompt = f"""Here is the product:

        Product Name: {product['name']}
        Product Description: {product['description']}
        Product Class: {product['class']}
        Category Hierachy: {product['category_hierarchy']}

        Describe the pros/cons/features of the product to satisfy the query: {query}

    """

    pros_cons = agent(product_prompt)
    agent.end()
    return pros_cons['responses']


def parse_decision(response):
    decision = response.split("\n")[-1]
    if 'LHS' in decision and 'RHS' not in decision:
        return 'LHS'
    elif 'RHS' in decision and 'LHS' not in decision:
        return 'RHS'
    else:
        return 'Neither'


def decide(query, product_lhs, product_rhs, pros_lhs, pros_rhs):
    instruction = [
        "You are a text agent trying to be helpful when evaluating two products.",
        f"Tell me which of the two products are most relevant based on the e-commerce search query: {query}"
        "You will see the pros/cons/features of the two products and you are to decide which product is more relevant to the query." # noqa
        f"Here is the first product, we'll name \"LHS\":\n\n {product_lhs['name']}",
        f"Here is the second product, we'll name \"RHS\":\n\n {product_rhs['name']}",
        f"Here are the pros/cons/features of the first product: {pros_lhs}",
        f"Here are the pros/cons/features of the second product: {pros_rhs}",
        f"Step by step, describe the information need behind the query: {query}",
        f"Now step-by-step describe which product (LHS or RHS) is more relevant to the query: {query}"
        f"Finally, conclude the last line with the product that is more relevant to the query.",
        "IMPORTANT for parsing, in addition, finish with a single line with either simply 'LHS' or 'RHS' with your decision."
    ]
    response = pv.generate("\n\n".join(instruction))
    decision = response.split("\n")[-1]
    if 'LHS' in decision:
        return 'LHS', response
    elif 'RHS' in decision:
        return 'RHS', response
    else:
        return 'Neither', response


def describe_and_decide(query, product_lhs, product_rhs):
    pros_lhs = describe(query, product_lhs)
    pros_rhs = describe(query, product_rhs)
    return decide(query, product_lhs, product_rhs, pros_lhs, pros_rhs)[0]


def title(query, product_lhs, product_rhs):
    instruction = f"""
        Which of these products is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS: {product_lhs['name']}
        Product RHS: {product_rhs['name']}

        Respond with just 'LHS' or 'RHS'
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def title_allow_neither(query, product_lhs, product_rhs):
    if product_lhs['name'] == product_rhs['name']:
        return 'Neither'
    instruction = f"""
        Neither product is more relevant to the query, unless given compelling evidence.

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS name: {product_lhs['name']}
            (remaining product attributes omited)
        Or
        Product RHS name: {product_rhs['name']}
            (remaining product attributes omited)
        Or
        Neither / Need more product attributes

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        Respond with just 'LHS - I am confident', 'RHS - I am confident', or 'Neither - not confident' with no other text. Respond 'Neither' if not enough evidence.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def title_w_desc_allow_neither(query, product_lhs, product_rhs):
    if product_lhs['name'] == product_rhs['name']:
        return 'Neither'
    instruction = f"""
        Neither product is more relevant to the query, unless given compelling evidence.

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS name: {product_lhs['name']}
        Product LHS description: {product_lhs['description']}
            (remaining product attributes omited)
        Or
        Product RHS name: {product_rhs['name']}
        Product RHS description: {product_rhs['description']}
            (remaining product attributes omited)
        Or
        Neither / Need more product attributes

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        Respond with just 'LHS - I am confident', 'RHS - I am confident', or 'Neither - not confident' with no other text. Respond 'Neither' if not enough evidence.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def class_allow_neither(query, product_lhs, product_rhs):
    if product_lhs['class'] == product_rhs['class']:
        return 'Neither'

    instruction = f"""
        Neither product is more relevant to the query, unless given compelling evidence.

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS class: {product_lhs['class']}
            (remaining product attributes omited)
        Or
        Product RHS class: {product_rhs['class']}
            (remaining product attributes omited)
        Or
        Neither / Need more product attributes

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        Respond with just 'LHS - I am confident', 'RHS - I am confident', or 'Neither - not confident' with no other text. Respond 'Neither' if not enough evidence.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def category_allow_neither(query, product_lhs, product_rhs):
    if product_lhs['category_hierarchy'] == product_rhs['category_hierarchy']:
        return 'Neither'

    instruction = f"""
        Neither product is more relevant to the query, unless given compelling evidence.

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS category hierarchy: {product_lhs['category_hierarchy']}
            (remaining product attributes omited)
        Or
        Product RHS category hierarchy: {product_rhs['category_hierarchy']}
            (remaining product attributes omited)
        Or
        Neither / Need more product attributes

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        Respond with just 'LHS - I am confident', 'RHS - I am confident', or 'Neither - not confident' with no other text. Respond 'Neither' if not enough evidence.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def desc_allow_neither(query, product_lhs, product_rhs):
    instruction = f"""
        Neither product is more relevant to the query, unless given compelling evidence.

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS description: {product_lhs['description']}
            (remaining product attributes omited)
        Or
        Product RHS description: {product_rhs['description']}
            (remaining product attributes omited)
        Or
        Neither / Need more product attributes

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        Respond with just 'LHS - I am confident', 'RHS - I am confident', or 'Neither - not confident' with no other text. Respond 'Neither' if not enough evidence.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def run_unanimous_ensemble(query, product_lhs, product_rhs, decision_fns):
    num_lhs = 0
    num_rhs = 0
    num_neither = 0
    # Present left as right, right as left
    # to deal with any weird biases for one or the other
    for decision_fn in decision_fns:
        decision = decision_fn(query, product_lhs, product_rhs)
        if decision == 'LHS':
            num_lhs += 1
        elif decision == 'RHS':
            num_rhs += 1
        else:
            return 'Neither'
        # Present left as right, right as left
        decision_reversed = decision_fn(query, product_rhs, product_lhs)
        if decision_reversed == 'LHS':
            num_rhs += 1
        elif decision_reversed == 'RHS':
            num_lhs += 1
        else:
            num_neither += 1
        total_decisions = num_lhs + num_rhs + num_neither

    print("Ensemble Complete")
    print(f"num_lhs: {num_lhs}, num_rhs: {num_rhs}, num_neither: {num_neither}")

    total_decisions = num_lhs + num_rhs + num_neither
    if num_lhs == total_decisions:
        return 'LHS'
    elif num_rhs == total_decisions:
        return 'RHS'
    else:
        return 'Neither'


def unanimous_ensemble_title(query, product_lhs, product_rhs):
    decision_fns = [title_allow_neither]
    return run_unanimous_ensemble(query, product_lhs, product_rhs, decision_fns)


def unanimous_ensemble_title_desc(query, product_lhs, product_rhs):
    decision_fns = [title_allow_neither, desc_allow_neither]
    return run_unanimous_ensemble(query, product_lhs, product_rhs, decision_fns)


def unanimous_ensemble_title_w_desc(query, product_lhs, product_rhs):
    decision_fns = [title_w_desc_allow_neither]
    return run_unanimous_ensemble(query, product_lhs, product_rhs, decision_fns)


def title_allow_neither2(query, product_lhs, product_rhs):
    instruction = f"""
        Youre playing a game. Youre evaluating the relevance of a product to a search query.

        You get:
           1  point for saying "I dont know"
           2  points if you pick the product most relevant (LHS or RHS below)
           -1 points for the wrong product name. So be careful! Choose LHS or RHS only when confident!

        Which of these product names (if either) is more relevant to the furniture e-commerce search query:

        Query: {query}

        Product LHS name: {product_lhs['name']}
            (remaining product attributes omited)
        Or
        Product RHS name: {product_rhs['name']}
            (remaining product attributes omited)
        Or
        I dont know

        Only respond 'LHS' or 'RHS' if you are confident in your decision

        To play respond with just 'LHS', 'RHS', or 'I dont know' with no other text.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def title_cot(query, product_lhs, product_rhs):
    if not query:
        raise ValueError("Query cannot be empty")
    instruction = f"""
        In one or two sentences, explain the information need behind this query: {query}.

        Then decide which of these products is more relevant to the query

        Product LHS: {product_lhs['name']}
        Product RHS: {product_rhs['name']}

        IMPORTANT: On a new line, your last line should just be 'LHS' or 'RHS',
        indicating the most relevant product, with no other text for easier parsing.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def title_cot2(query, product_lhs, product_rhs):
    if not query:
        raise ValueError("Query cannot be empty")
    instruction = f"""
        In one or two sentences, explain the information need behind this query: {query}.

        In one or two sentences, describe each of these products characteristics:
        Product LHS: {product_lhs['name']}
        Product RHS: {product_rhs['name']}

        Next, describe in a few sentences the pros/cons of these products for the query.

        Then decide which of these products is more relevant to the query

        IMPORTANT: On a new line, your last line should just be 'LHS' or 'RHS',
        indicating the most relevant product, with no other text for easier parsing.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def all_cot(query, product_lhs, product_rhs):
    if not query:
        raise ValueError("Query cannot be empty")
    instruction = f"""
        In one or two sentences, explain the information need behind this query: {query}.

        In one or two sentences, describe each of these products characteristics:
        Product LHS: {product_lhs['name']}
           Description: {product_lhs['description']}
                 Class: {product_lhs['class']}
    Category Hierarchy: {product_lhs['category_hierarchy']}
        Product RHS: {product_rhs['name']}
           Description: {product_rhs['description']}
                 Class: {product_rhs['class']}
    Category Hierarchy: {product_rhs['category_hierarchy']}


        Next, describe in a few sentences the pros/cons of these products for the query.

        Then decide which of these products is more relevant to the query by simply
        placing 'LHS' or 'RHS' on the last line.
    """
    response = pv.generate(instruction)
    return parse_decision(response)


def all_fns():
    return [
        title,
        title_allow_neither2,
        title_cot,
        title_cot2,
        all_cot,
        unanimous_ensemble_title,
        unanimous_ensemble_title_desc,
        unanimous_ensemble_title_w_desc,
        title_allow_neither,
        title_w_desc_allow_neither,
        class_allow_neither,
        category_allow_neither,
        desc_allow_neither
    ]
