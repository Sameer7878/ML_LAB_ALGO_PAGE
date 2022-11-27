const express =require('express');
const app = express();
const port = 5000;
app.post('/upload', (req, res, next) => {
    let uploadFile = req.files.file
    const fileName = req.files.file.name
    uploadFile.mv(
      `${__dirname}/public/files/${fileName}`,
      function (err) {
        if (err) {
          return res.status(500).send(err)
        }
        res.json({
          file: `public/${req.files.file.name}`,
        })
      },
    )
  })
  app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
  })