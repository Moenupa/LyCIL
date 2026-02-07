# Contributing to LyCIL

Everyone is welcome to contribute, and we value everybody's contribution. Code
 contributions are not the only way to help the community. Answering questions, helping
 others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
 about the awesome projects it made possible, shout out on Twitter every time it has 
 helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our [code of conduct](CODE_OF_CONDUCT.md).

**This guide was heavily inspired by [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to LyCIL:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new models.
* Contribute to the examples or to the documentation.

If you don't know where to start, there is a special [Good First
Issue][1st-issue] listing. It will give you a list of
open issues that are beginner-friendly and help you start contributing to open-source.
 The best way to do that is to open a Pull Request and link it to the issue that 
 you'd like to work on. We try to give priority to opened PRs as we can easily 
 track the progress of the fix, and if the contributor does not have time anymore, 
 someone else can take the PR over.

For something slightly more challenging, you can also take a look at the [Good Second Issue][2nd-issue] list. In general though, if you feel like you know what you're doing, go for it and we'll help you get there! 🚀

> All contributions are equally valuable to the community. 🥰

[1st-issue]: https://github.com/Moenupa/LyCIL/contribute
[2nd-issue]: https://github.com/Moenupa/LyCIL/labels/Good%20Second%20Issue
[lycil]: https://github.com/Moenupa/LyCIL
[lycil-fork]: https://github.com/Moenupa/LyCIL/fork

### Style guide

LyCIL follows the [Google Python Style Guide][google-python-styleguide] and 
 [PyTorch Docstring Style Guide][torch-docstring-styleguide].

[google-python-styleguide]: https://google.github.io/styleguide/pyguide.html
[torch-docstring-styleguide]: https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines

### Create a Pull Request

1. Fork the [repository][lycil] by clicking on the [Fork][lycil-fork] button on the repository's page. This creates a copy of the code under your GitHub user account.
2. Clone your fork to your local disk, and add the base repository as a remote:
    ```bash
    git clone git@github.com:<your Github handle>/LyCIL.git
    cd LyCIL
    git remote add upstream https://github.com/Moenupa/LyCIL.git
    ```

3. Create a new branch to hold your development changes:
    ```bash
    git checkout -b feat/describe_your_changes
    ```

4. Set up a development environment by running the following command in a virtual environment:
    ```bash
    uv sync --extra cuda --dev  # or --extra npu for NPU environment
    ```

5. Check code before commit:
    ```bash
    make install
    make style && make style-check
    make quality
    make test
    ```

6. Submit changes:
    ```bash
    git add .
    git commit -m "commit message"
    git fetch upstream
    git rebase upstream/main
    git push -u origin feat/describe_your_changes
    ```

7. Create a pull request from your branch `feat/describe_your_changes` at [origin repo][lycil].