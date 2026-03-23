IMAGE  = credit-risk
PORT   = 8000

# ── Default target ─────────────────────────────────────────────────────────────
# Single command: build image (runs prepare_data inside) then start the API
.PHONY: run
run: build
	docker run --rm -p $(PORT):8000 --name $(IMAGE) $(IMAGE)

# ── Build ──────────────────────────────────────────────────────────────────────
.PHONY: build
build:
	docker build -t $(IMAGE) .

# ── Dev: run with live reload (mounts local files, no rebuild needed) ──────────
.PHONY: dev
dev:
	docker run --rm -p $(PORT):8000 \
		-v $(PWD)/app.py:/app/app.py \
		-v $(PWD)/artifacts:/app/artifacts \
		--name $(IMAGE)-dev \
		$(IMAGE) \
		uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# ── Test: hit both predict endpoints with CUST_0002 ───────────────────────────
.PHONY: test
test:
	@echo "\n--- POST /predict (raw) ---"
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"customer_id":"CUST_0002","txn_count":2,"total_debit":-650.0,"total_credit":1800.0,"avg_amount":575.0,"kw_rent":1,"kw_netflix":0,"kw_tesco":0,"kw_payroll":1,"kw_bonus":0}' \
		| python3 -m json.tool
	@echo "\n--- POST /predict/scaled (recommended) ---"
	curl -s -X POST http://localhost:$(PORT)/predict/scaled \
		-H "Content-Type: application/json" \
		-d '{"customer_id":"CUST_0002","txn_count":2,"total_debit":-650.0,"total_credit":1800.0,"avg_amount":575.0,"kw_rent":1,"kw_netflix":0,"kw_tesco":0,"kw_payroll":1,"kw_bonus":0}' \
		| python3 -m json.tool
	@echo "\n--- GET /health ---"
	curl -s http://localhost:$(PORT)/health | python3 -m json.tool

# ── Stop running container ─────────────────────────────────────────────────────
.PHONY: stop
stop:
	docker stop $(IMAGE) $(IMAGE)-dev 2>/dev/null || true

# ── Clean up image ─────────────────────────────────────────────────────────────
.PHONY: clean
clean: stop
	docker rmi $(IMAGE) 2>/dev/null || true

.PHONY: help
help:
	@echo ""
	@echo "  make run     build image and start API on localhost:$(PORT)  one command for all"
	@echo "  make build   build Docker image only"
	@echo "  make dev     run with live reload no rebuild on code change"
	@echo "  make test    hit both predict endpoints + health check"
	@echo "  make stop    stop running container"
	@echo "  make clean   stop + remove image"
	@echo ""