import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
        name="colvar",
        author="Kirill Zinovjev",
        author_email="kzinovjev@gmail.com",
        description="Recursive collective variables for MD simulations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/kzinovjev/rpcv",
        packages=["colvar"],
        version="0.1"
)