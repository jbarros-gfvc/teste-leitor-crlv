import base64
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import segno
from PIL import Image

from teste_leitor_crlv.analise_local import (
    ValidacaoLocalRequest,
    validar_localmente_json,
)
from teste_leitor_crlv.analise_ocr import (
    OcrCrlvRequest,
    analisar_crlv_com_posicao,
    corrigir_placa,
    delimitar_documento_pelos_textos,
    dentro_da_area,
    ocr_crlv_json,
)
from teste_leitor_crlv.analise_paddleocr import (
    OcrPaddleRequest,
    analisar_crlv_com_paddleocr,
    ocr_paddle_json,
)
from teste_leitor_crlv.analise_paralela import (
    AnaliseRapidaRequest,
    analisar_rapido_json,
    orquestrar_corrida_paralela,
)
from teste_leitor_crlv.analise_vio import (
    ConsultaVioRequest,
    consultar_json,
)
from teste_leitor_crlv.enquadramento import (
    MetadadosEnquadramento,
    otimizar_enquadramento_crlv,
)
from teste_leitor_crlv.qrcode_utils import (
    extrair_bytes_qrcode,
    gerar_qrcode_base64,
)


class TestVioAndLocalChecker(unittest.TestCase):
    def test_binary_roundtrip(self):
        dados_originais = b"\x00\xff\xfe\x01\x02\xaa\xbb\xcc\xdd\xee"
        qr = segno.make(dados_originais)
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=5)
        imagem_bytes = buf.getvalue()

        bytes_extraidos = extrair_bytes_qrcode(imagem_bytes)
        self.assertEqual(bytes_extraidos, dados_originais)

        b64_reconstituido = gerar_qrcode_base64(bytes_extraidos)
        self.assertIsInstance(b64_reconstituido, str)

    @patch("teste_leitor_crlv.analise_vio.VioClient")
    def test_consultar_json_com_imagem_base64(self, mock_client_cls):
        mock_instance = mock_client_cls.return_value
        mock_instance.read_qrcode.return_value = {
            "id": "cnh_123",
            "status": "completed",
            "document": {"tipo": "CNH"},
        }

        qr = segno.make(b"\x01\x02\x03\x04\x05")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=5)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        req = ConsultaVioRequest(
            imagem_base64=f"data:image/png;base64,{b64_img}",
            wait=True,
            incluir_qrcode_base64=True,
        )
        res = consultar_json(req)
        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "integrador-vio")
        self.assertEqual(res["payload"]["bytes_extraidos_total"], 5)
        self.assertEqual(res["payload"]["dados_vio"]["id"], "cnh_123")
        self.assertIn("qrcode_reconstituido_base64", res["payload"])

    @patch("teste_leitor_crlv.analise_local.DigitalDocumentChecker")
    def test_validar_localmente_json(self, mock_checker_cls):
        mock_instance = mock_checker_cls.return_value
        mock_res = mock_instance.parse_base64.return_value
        mock_res.to_dict.return_value = {
            "document_type": "cnh",
            "status": {
                "is_parsed": True,
                "is_authentic": True,
                "is_valid": True,
            },
            "fields": {"nome": "TESTE LOCAL", "cpf": "12345678900"},
        }

        qr = segno.make(b"conteudo-qr-teste")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=5)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        req = ValidacaoLocalRequest(
            imagem_base64=b64_img,
            include_raw=False,
        )
        res = validar_localmente_json(req)
        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "digital-document-checker")
        self.assertEqual(res["payload"]["document_type"], "cnh")
        self.assertTrue(res["payload"]["status"]["is_authentic"])

    def test_dentro_da_area(self):
        limites = (0.1, 0.1, 0.3, 0.3)
        self.assertTrue(dentro_da_area(0.2, 0.2, limites, margem=0.10))
        self.assertFalse(dentro_da_area(0.6, 0.6, limites, margem=0.10))

    def test_corrigir_placa(self):
        self.assertEqual(corrigir_placa("0BC1D2O"), "OBC1D20")
        self.assertEqual(corrigir_placa("BRA2E19"), "BRA2E19")

    def test_delimitar_documento(self):
        mock_res = [
            ([[100, 200], [400, 200], [400, 300], [100, 300]], "TEXTO", 0.9),
            ([[150, 500], [600, 500], [600, 800], [150, 800]], "RENAVAM", 0.8),
        ]
        x_min, y_min, x_max, y_max = delimitar_documento_pelos_textos(mock_res)
        self.assertEqual(x_min, 100)
        self.assertEqual(y_min, 200)
        self.assertEqual(x_max, 600)
        self.assertEqual(y_max, 800)

    def test_analisar_crlv_mock_reader(self):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[200, 50], [600, 50], [600, 100], [200, 100]], "REPUBLICA FEDERATIVA DO BRASIL", 0.99),
            ([[100, 220], [600, 220], [600, 300], [100, 300]], "CERTIFICADO DE REGISTRO E LICENCIAMENTO", 0.99),
            ([[100, 400], [350, 400], [350, 480], [100, 480]], "RENAVAM 12345678901", 0.99),
            ([[100, 520], [250, 520], [250, 600], [100, 600]], "PLACA ABC1D23", 0.99),
            ([[300, 520], [500, 520], [500, 600], [300, 600]], "EXERCICIO 2024", 0.99),
            ([[100, 750], [400, 750], [400, 850], [100, 850]], "NUMERO DO CRV 1234567890", 0.99),
        ]

        dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        res = analisar_crlv_com_posicao(dummy_img, reader=mock_reader, auto_orientar=False)

        self.assertTrue(res["formato_crlv_valido"])
        self.assertEqual(res["placa"], "ABC1D23")
        self.assertEqual(res["renavam"], "12345678901")
        self.assertEqual(res["total_ancoras_ok"], "6/6")
        self.assertIn("area_documento_detectada", res)

    @patch("teste_leitor_crlv.analise_ocr.obter_reader_ocr")
    def test_endpoint_ocr_crlv_json(self, mock_get_reader):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[200, 50], [600, 50], [600, 100], [200, 100]], "REPUBLICA FEDERATIVA", 0.99),
            ([[100, 220], [600, 220], [600, 300], [100, 300]], "CERTIFICADO DE REGISTRO", 0.99),
            ([[100, 400], [350, 400], [350, 480], [100, 480]], "RENAVAM 12345678901", 0.99),
            ([[100, 520], [250, 520], [250, 600], [100, 600]], "PLACA BRA2E19", 0.99),
            ([[300, 520], [500, 520], [500, 600], [300, 600]], "EXERCICIO 2024", 0.99),
            ([[100, 750], [400, 750], [400, 850], [100, 850]], "NUMERO DO CRV 1234567890", 0.99),
        ]
        mock_get_reader.return_value = mock_reader

        img = Image.new("RGB", (1000, 1000), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        req = OcrCrlvRequest(imagem_base64=b64, margem_erro=0.10)
        res = ocr_crlv_json(req)

        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "easyocr")
        self.assertEqual(res["placa"], "BRA2E19")
        self.assertEqual(res["payload"]["renavam"], "12345678901")

    def test_paddleocr_mock_reader(self):
        mock_reader = MagicMock()
        mock_reader.predict.return_value = [
            {
                "rec_polys": [
                    np.array([[200, 50], [600, 50], [600, 100], [200, 100]]),
                    np.array([[100, 220], [600, 220], [600, 300], [100, 300]]),
                    np.array([[100, 400], [350, 400], [350, 480], [100, 480]]),
                    np.array([[100, 520], [250, 520], [250, 600], [100, 600]]),
                    np.array([[300, 520], [500, 520], [500, 600], [300, 600]]),
                    np.array([[100, 750], [400, 750], [400, 850], [100, 850]]),
                ],
                "rec_texts": [
                    "REPUBLICA FEDERATIVA DO BRASIL",
                    "CERTIFICADO DE REGISTRO E LICENCIAMENTO",
                    "RENAVAM 00381050920",
                    "PLACA BWY2C88",
                    "EXERCICIO 2026",
                    "NUMERO DO CRV 264901016857",
                ],
                "rec_scores": [0.99, 0.98, 0.99, 0.99, 0.99, 0.99],
            }
        ]

        dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        res = analisar_crlv_com_paddleocr(dummy_img, reader=mock_reader, usar_modelo_local=True)

        self.assertTrue(res["formato_crlv_valido"])
        self.assertEqual(res["placa"], "BWY2C88")
        self.assertEqual(res["renavam"], "00381050920")
        self.assertEqual(res["total_ancoras_ok"], "6/6")
        self.assertTrue(res["base_treinamento_local_ativa"])
        self.assertIn("local_treinado", res["motor"])

    @patch("teste_leitor_crlv.analise_paddleocr.obter_reader_paddle")
    def test_endpoint_paddleocr_json(self, mock_get_paddle):
        mock_reader = MagicMock()
        mock_reader.predict.return_value = [
            {
                "rec_polys": [
                    np.array([[200, 50], [600, 50], [600, 100], [200, 100]]),
                    np.array([[100, 220], [600, 220], [600, 300], [100, 300]]),
                    np.array([[100, 400], [350, 400], [350, 480], [100, 480]]),
                    np.array([[100, 520], [250, 520], [250, 600], [100, 600]]),
                    np.array([[300, 520], [500, 520], [500, 600], [300, 600]]),
                    np.array([[100, 750], [400, 750], [400, 850], [100, 850]]),
                ],
                "rec_texts": [
                    "REPUBLICA FEDERATIVA",
                    "CERTIFICADO DE REGISTRO",
                    "RENAVAM 00381050920",
                    "PLACA BWY2C88",
                    "EXERCICIO 2026",
                    "NUMERO DO CRV 264901016857",
                ],
                "rec_scores": [0.99, 0.98, 0.99, 0.99, 0.99, 0.99],
            }
        ]
        mock_get_paddle.return_value = mock_reader

        img = Image.new("RGB", (1000, 1000), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        req = OcrPaddleRequest(imagem_base64=b64, margem_erro=0.10)
        res = ocr_paddle_json(req)

        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "paddleocr")
        self.assertEqual(res["placa"], "BWY2C88")
        self.assertEqual(res["payload"]["renavam"], "00381050920")
        self.assertIn("enquadramento", res["payload"])

    def test_enquadramento_sem_qr(self):
        img_np = np.ones((600, 800, 3), dtype=np.uint8) * 255
        crop_rgb, meta = otimizar_enquadramento_crlv(img_np)
        self.assertIsInstance(meta, MetadadosEnquadramento)
        self.assertEqual(meta.dimensoes_originais, (800, 600))
        self.assertEqual(crop_rgb.shape, (600, 800, 3))

    def test_enquadramento_com_qr(self):
        # Cria imagem com QR Code posicionado a direita
        qr = segno.make(b"0123456789ABCDEF")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=4)
        qr_pil = Image.open(buf).convert("RGB")

        # Imagem grande 1200x800 com QR Code no canto direito
        base_img = Image.new("RGB", (1200, 800), color="white")
        base_img.paste(qr_pil, (800, 300))
        img_np = np.array(base_img)

        crop_rgb, meta = otimizar_enquadramento_crlv(img_np)
        self.assertTrue(meta.crop_aplicado)
        self.assertEqual(meta.metodo, "qrcode_vio")
        self.assertTrue(meta.qrcode_info["bytes_length"] > 0)
        self.assertLessEqual(crop_rgb.shape[1], 1200)
        self.assertLessEqual(crop_rgb.shape[0], 800)

    @patch("teste_leitor_crlv.analise_paralela._executar_digital_checker")
    @patch("teste_leitor_crlv.analise_paralela._executar_easyocr")
    @patch("teste_leitor_crlv.analise_paralela._executar_paddleocr")
    def test_corrida_paralela_vencedor_checker(self, mock_paddle, mock_easy, mock_checker):
        mock_checker.return_value = {
            "sucesso": True,
            "analisador": "digital_document_checker",
            "tempo_ms": 12.5,
            "dados": {"status": {"is_parsed": True}, "document": {"numero_espelho": "123"}},
        }
        mock_easy.return_value = {"sucesso": False, "analisador": "easyocr", "tempo_ms": 450.0}
        mock_paddle.return_value = {"sucesso": False, "analisador": "paddleocr", "tempo_ms": 300.0}

        img_bytes = b"fake_crlv_bytes"
        res = orquestrar_corrida_paralela(
            imagem_bytes=img_bytes,
            margem_erro=0.10,
            garantir_treino_paddle=True,
        )

        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "digital_document_checker")
        self.assertTrue(res["payload"]["paddle_coletando_em_background"])
        self.assertEqual(res["payload"]["dados"]["document"]["numero_espelho"], "123")

    @patch("teste_leitor_crlv.analise_paralela.orquestrar_corrida_paralela")
    def test_endpoint_analisar_rapido_json(self, mock_orquestrador):
        mock_orquestrador.return_value = {
            "sucesso": True,
            "validador": "paddleocr",
            "arquivo": "base64_upload.png",
            "placa": "ABC1D23",
            "payload": {
                "tempo_resposta_ms": 180.2,
                "paddle_coletando_em_background": False,
                "dados": {"placa": "ABC1D23", "renavam": "12345678901"},
            },
        }
        b64 = base64.b64encode(b"dummy_image").decode("ascii")
        req = AnaliseRapidaRequest(imagem_base64=b64, garantir_treino_paddle=True)
        res = analisar_rapido_json(req)

        self.assertTrue(res["sucesso"])
        self.assertEqual(res["validador"], "paddleocr")
        self.assertEqual(res["placa"], "ABC1D23")


if __name__ == "__main__":
    unittest.main()
