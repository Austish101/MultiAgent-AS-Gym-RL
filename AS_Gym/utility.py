
def validation(question, num_only, answer0="None", answer1="None", answer2="None", answer3="None", answer4="None"):
    # validation for 2-3 options, or numerical check, repeat question if invalid
    # answers must be uppercase if alphanumeric
    valid = False
    while not valid:
        response = input(question)
        if num_only and response.isnumeric():
            return response

        if answer0 != "None":
            if answer0 == response:
                return 0
            elif answer0.isalpha() and response.isalpha():
                if answer0.lower() == response.lower():
                    return 0

        if answer1 != "None":
            if answer1 == response:
                return 1
            elif answer1.isalpha() and response.isalpha():
                if answer1.lower() == response.lower():
                    return 1

        if answer2 != "None":
            if answer2 == response:
                return 2
            elif answer2.isalpha() and response.isalpha():
                if answer2.lower() == response.lower():
                    return 2

        if answer3 != "None":
            if answer3 == response:
                return 3
            elif answer3.isalpha() and response.isalpha():
                if answer3.lower() == response.lower():
                    return 3

        if answer4 != "None":
            if answer4 == response:
                return 4
            elif answer4.isalpha() and response.isalpha():
                if answer4.lower() == response.lower():
                    return 4

        if (response == "Q") or (response == "q"):
            print("Exiting program...")
            exit()
        else:
            if num_only:
                print("Invalid response, please enter a number")
            elif answer4 != "None":
                print("Invalid response, please enter", answer0, "or", answer1, "or", answer2, "or", answer3, "or",
                      answer4)
            elif answer3 != "None":
                print("Invalid response, please enter", answer0, "or", answer1, "or", answer2, "or", answer3)
            elif answer2 != "None":
                print("Invalid response, please enter", answer0, "or", answer1, "or", answer2)
            else:
                print("Invalid response, please enter", answer0, "or", answer1)
