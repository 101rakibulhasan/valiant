const prompt = `Hi`;

async function Test() {
    let response = await fetch("http://127.0.0.1:45123/completion", {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            n_predict: 512,
        })
    })
    console.log((await response.json()))
}

Test()